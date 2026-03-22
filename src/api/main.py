"""main.py

Σηκώνει το FastAPI app και συνδέει:
- schemas
- preprocessing
- configs
- frozen model
- structured JSON logging

Εκθέτει τα endpoints:
- GET /health
- GET /metadata
- POST /predict
- POST /predict_by_id

Εφαρμόζει:
- frozen threshold logic
- consistent serving pipeline
- global exception handling
- request-level και prediction-level observability
"""

from typing import Any
import time
import uuid # Χρησιμοποιείται για τη δημιουργία μοναδικού request ID για κάθε HTTP request.

import xgboost as xgb
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# Relative imports από το ίδιο package
from .logging_config import configure_logging, get_logger
from .model_loader import (
    get_model_path,
    get_model_version,
    load_demo_test_df,
    load_feature_schema,
    load_model,
    load_model_metadata,
    load_threshold_config,
)
from .preprocess import prepare_single_payload
from .schemas import (
    MetadataResponse,
    PredictByIdRequest,
    PredictRequest,
    PredictResponse,
)

# Logging initialization
configure_logging()
logger = get_logger()

# FastAPI application object
app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="0.1.0",
    description="Frozen Week 17 inference API for the XGBoost champion model.",
)


# helper function για να είναι ο κώδικας version-compatible
def _to_dict(pydantic_model: Any) -> dict[str, Any]:
    """Convert a Pydantic model to a plain Python dict (Pydantic v1/v2 compatible)."""
    if hasattr(pydantic_model, "model_dump"):
        return pydantic_model.model_dump()
    return pydantic_model.dict()

 # Αυτή η function επιστρέφει το request_id από το request object.
def _get_request_id(request: Request) -> str | None:
    """Safely read request_id from request.state if present."""
    return getattr(request.state, "request_id", None)


# Αυτή η function φτιάχνει JSON error response.
def _json_error_response(
    request_id: str | None,
    status_code: int,
    content: dict[str, Any],
) -> JSONResponse:
    """Build JSON error response and attach X-Request-ID when available."""
    response = JSONResponse(status_code=status_code, content=content)
    if request_id:
        response.headers["X-Request-ID"] = request_id
    return response


# Αυτή η function είναι ο πυρήνας του scoring pipeline.
def _score_payload(payload_dict: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """
    Shared scoring helper.

    Measures end-to-end serving latency for:
    - preprocessing
    - feature alignment
    - model scoring
    """
    start = time.perf_counter()

    feature_schema = load_feature_schema()
    threshold_cfg = load_threshold_config()
    model = load_model()

    X = prepare_single_payload(payload_dict, feature_schema)
    threshold_used = float(threshold_cfg["threshold"])

    # Preferred path for XGBoost models: build DMatrix with explicit feature names
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        dmatrix = xgb.DMatrix(X.values, feature_names=list(X.columns))
        fraud_probability = float(booster.predict(dmatrix)[0])
    else:
        if not hasattr(model, "predict_proba"):
            raise RuntimeError("Loaded model does not support predict_proba().")
        fraud_probability = float(model.predict_proba(X)[0][1])

    # Threshold logic
    predicted_label = "fraud" if fraud_probability >= threshold_used else "legit"
    prediction_latency_ms = round((time.perf_counter() - start) * 1000, 3)

    # structured prediction output
    result = {
        "model_version": get_model_version(),
        "threshold_policy": threshold_cfg["policy_name"],
        "threshold_used": threshold_used,
        "fraud_probability": fraud_probability,
        "predicted_label": predicted_label,
    }

    return result, prediction_latency_ms


# HTTP middleware. Middleware σημαίνει ότι εκτελείται γύρω από κάθε request: πριν φτάσει στο endpoint και αφού γυρίσει response
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Request-level logging middleware.

    Logs:
    - request_id
    - method
    - path
    - final status code
    - total request latency

    Also logs a dedicated event if a failure happens before a response object exists.
    """
    request_id = uuid.uuid4().hex # Το .hex δίνει compact hexadecimal string χωρίς παύλες
    request.state.request_id = request_id # Το request.state είναι χώρος όπου μπορείς να αποθηκεύσεις custom data για το συγκεκριμένο request.

    # Ξεκινάς μέτρηση συνολικού request latency.
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        latency_ms = round((time.perf_counter() - start) * 1000, 3)

        # Log failure
        logger.warning(
            "request_failed_before_response",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            latency_ms=latency_ms,
        )
        raise # Re-raise the exception to be handled by global exception handlers

    # Success path latency. Αν δεν έγινε exception, μετράς τη συνολική διάρκεια του request
    latency_ms = round((time.perf_counter() - start) * 1000, 3)
    response.headers["X-Request-ID"] = request_id # Βάζεις το request ID στο response header

    # Log successful request
    logger.info(
        "request_completed",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=latency_ms,
    )

    return response # Επιστρέφεις το τελικό HTTP response.


# Αυτός είναι ο global handler για validation errors. Τέτοια errors συμβαίνουν όταν το request body δεν ταιριάζει με το schema.
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Global handler for FastAPI/Pydantic validation errors (422).
    """
    request_id = _get_request_id(request)

    # Log validation error
    logger.warning(
        "validation_error",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=422,
        errors_count=len(exc.errors()), # Πόσα validation problems βρέθηκαν
        errors=exc.errors(), # Η πλήρης λίστα validation errors.
    )

    return _json_error_response(
        request_id=request_id,
        status_code=422,
        content={"detail": exc.errors()},
    )


# Αυτός ο handler πιάνει explicit HTTP errors που σηκώνεις εσύ στον κώδικα.
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Global handler for explicit HTTP errors.
    """
    request_id = _get_request_id(request)

    # Log HTTP error
    logger.warning(
        "http_error",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=exc.status_code,
        detail=exc.detail,
    )

    return _json_error_response(
        request_id=request_id,
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


# Αυτός πιάνει ό,τι άλλο exception δεν πιάστηκε από τους άλλους handlers.
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Global handler for unexpected server-side exceptions.
    """
    request_id = _get_request_id(request)

    logger.exception(
        "unhandled_exception",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        error_type=type(exc).__name__,
    )

    return _json_error_response(
        request_id=request_id,
        status_code=500, # Return generic 500
        content={"detail": "Internal server error"},
    )


# Endpoint /health
@app.get("/health")
def health():
    return {"status": "ok"}


# Endpoint /metadata
@app.get("/metadata", response_model=MetadataResponse)
def metadata(request: Request):
    """
    Return serving metadata for the currently deployed model.

    Acts as a mini model-registry / provenance view for:
    - model artifact
    - threshold policy
    - schema version
    - training/build metadata
    """
    feature_schema = load_feature_schema()
    threshold_cfg = load_threshold_config()
    model_metadata = load_model_metadata()

    response = MetadataResponse(
        model_version=model_metadata["model_version"],
        model_artifact_path=str(get_model_path()),
        git_commit=model_metadata["git_commit"],
        train_date=model_metadata["train_date"],
        threshold_policy=threshold_cfg["policy_name"],
        threshold_policy_version=threshold_cfg["policy_version"],
        threshold_used=float(threshold_cfg["threshold"]),
        schema_version=feature_schema["schema_version"],
        raw_input_features=feature_schema["raw_input_features"],
        engineered_features=feature_schema["engineered_features"],
        model_features=feature_schema["model_features"],
        training_data_reference=model_metadata.get("training_data_reference"),
        training_target=model_metadata.get("training_target"),
        framework=model_metadata.get("framework"),
        task=model_metadata.get("task"),
    )

    logger.info(
        "metadata_requested",
        request_id=_get_request_id(request),
        endpoint="/metadata",
        model_version=response.model_version,
        git_commit=response.git_commit,
        threshold_policy=response.threshold_policy,
        threshold_policy_version=response.threshold_policy_version,
        schema_version=response.schema_version,
    )

    return response


# Endpoint /predict (κύριο endpoint για scoring)
@app.post("/predict", response_model=PredictResponse, response_model_exclude_none=True)
def predict(payload: PredictRequest, request: Request):
    payload_dict = _to_dict(payload)
    result, prediction_latency_ms = _score_payload(payload_dict)

    # Structured prediction log
    logger.info(
        "prediction_scored",
        request_id=_get_request_id(request),
        endpoint="/predict",
        prediction_latency_ms=prediction_latency_ms,
        fraud_probability=result["fraud_probability"],
        predicted_label=result["predicted_label"],
        threshold_used=result["threshold_used"],
        threshold_policy=result["threshold_policy"],
        model_version=result["model_version"],
    )

    return PredictResponse(**result) # Το **result κάνει dictionary unpacking.


# Endpoint /predict_by_id (demo/testing endpoint. Αντί να παίρνει raw features, παίρνει row_id.)
@app.post("/predict_by_id", response_model=PredictResponse, response_model_exclude_none=True)
def predict_by_id(payload: PredictByIdRequest, request: Request):
    demo_df = load_demo_test_df()
    feature_schema = load_feature_schema()
    target_column = feature_schema["target_column"]
    raw_input_features = feature_schema["raw_input_features"]

    # φιλτράρισμα στο DataFrame για να βρεις τη γραμμή με το συγκεκριμένο row_id
    matched = demo_df.loc[demo_df["row_id"] == payload.row_id]
    if matched.empty:
        raise HTTPException(status_code=404, detail=f"row_id {payload.row_id} not found")

    row = matched.iloc[0]

    # Rebuild payload from canonical raw inputs only,
    # so demo scoring uses the exact same serving pipeline as /predict
    rebuilt_payload = {col: float(row[col]) for col in raw_input_features}

    result, prediction_latency_ms = _score_payload(rebuilt_payload)
    result["row_id"] = int(payload.row_id)
    result["true_label"] = int(row[target_column])

    # Log prediction event
    logger.info(
        "prediction_scored",
        request_id=_get_request_id(request),
        endpoint="/predict_by_id",
        prediction_latency_ms=prediction_latency_ms,
        fraud_probability=result["fraud_probability"],
        predicted_label=result["predicted_label"],
        threshold_used=result["threshold_used"],
        threshold_policy=result["threshold_policy"],
        model_version=result["model_version"],
        row_id=result["row_id"],
        true_label=result["true_label"],
    )

    return PredictResponse(**result) # typed response μέσω Pydantic model
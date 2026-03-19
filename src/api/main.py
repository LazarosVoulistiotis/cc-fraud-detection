"""Το main.py είναι το αρχείο που:
    σηκώνει το FastAPI app
    συνδέει schemas, preprocessing, configs και model
    εκθέτει τα 4 endpoints
    εφαρμόζει το frozen threshold logic
    κρατά consistent serving pipeline
    διαχειρίζεται σωστά τα HTTP errors
Με πολύ απλά λόγια:
    είναι ο controller/orchestrator του Week 17 inference API σου"""

# FastAPI Χρησιμοποιείται για να δημιουργήσεις το application object. HTTPException Χρησιμοποιείται όταν θέλεις να επιστρέψεις HTTP error response με συγκεκριμένο status code.
from fastapi import FastAPI, HTTPException
import xgboost as xgb

# functions από το model_loader.py Το . μπροστά σημαίνει relative import από το ίδιο package (src/api)
from .model_loader import (
    get_model_version, # επιστρέφει π.χ. "xgb_final"
    load_demo_test_df, # φορτώνει το frozen test_with_row_id.csv
    load_feature_schema, # φορτώνει το feature_schema.json
    load_model, # φορτώνει το .joblib μοντέλο
    load_threshold_config, # φορτώνει το threshold.json
)
# function από το preprocess.py που μετατρέπει το raw input payload σε dataframe με σωστή feature σειρά και συμπληρωμένα engineered features
from .preprocess import prepare_single_payload
# Pydantic models για validation και documentation των API endpoints
from .schemas import (
    MetadataResponse, # structured output του /metadata
    PredictByIdRequest, # validation για /predict_by_id
    PredictRequest, # validation για /predict
    PredictResponse, # structured output των prediction endpoints
)

# Δημιουργία FastAPI app object. Αυτό είναι το object που θα “δει” το Uvicorn όταν κάνεις: uvicorn src.api.main:app --reload (Uvicorn is a lightning-fast ASGI web server implementation for Python, designed to run asynchronous frameworks like FastAPI and Django)
app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="0.1.0",
    description="Frozen Week 17 inference API for the XGBoost champion model."
)

# Helper function. Το _ στην αρχή είναι σύμβαση που δείχνει: δεν είναι public API function, είναι internal utility. Ο ρόλος της είναι να μετατρέπει ένα Pydantic model object σε Python dict.
def _to_dict(pydantic_model):
    if hasattr(pydantic_model, "model_dump"): # από Pydantic v2 και μετά, το model_dump() είναι ο νέος τρόπος να μετατρέψεις ένα Pydantic model σε dict. Αν δεν υπάρχει, fallback στο dict() που ήταν ο τρόπος σε Pydantic v1.
        return pydantic_model.model_dump()
    return pydantic_model.dict() # "δώσε μου το request model σαν απλό Python dictionary, ανεξάρτητα από version style"

# Helper function. Αυτή είναι η καρδιά του scoring logic. Δέχεται ένα payload σε μορφή Python dict, επιστρέφει το prediction result ως dict
def _score_payload(payload_dict: dict) -> dict:
    feature_schema = load_feature_schema() # φορτώνεις το frozen schema
    threshold_cfg = load_threshold_config() # φορτώνεις το threshold policy config
    model = load_model() # φορτώνεις το trained model object
    # Prepare model input: παίρνεις το raw payload, το περνάς από preprocessing, παίρνεις one-row aligned DataFrame. Άρα το X είναι πια έτοιμο για το model.
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
        
    predicted_label = "fraud" if fraud_probability >= threshold_used else "legit" # business rule. Αν η πιθανότητα είναι πάνω ή ίση με το threshold: predicted_label είναι "fraud", αλλιώς "legit".
    # επιστρέφεις dict με το structured output του prediction
    return {
        "model_version": get_model_version(),
        "threshold_policy": threshold_cfg["policy_name"],
        "threshold_used": threshold_used,
        "fraud_probability": fraud_probability,
        "predicted_label": predicted_label,
    }

# Endpoint 1: /health (όταν έρθει GET request στο /health, κάλεσε τη function health)
@app.get("/health")
def health():
    return {"status": "ok"}

# Endpoint 2: /metadata (όταν έρθει GET request στο /metadata, κάλεσε τη function metadata, που επιστρέφει structured metadata για το μοντέλο και τα features)
@app.get("/metadata", response_model=MetadataResponse)
def metadata():
    try: # μέσα σε try για να μετατρέψεις εσωτερικά errors σε HTTPException(500)
        feature_schema = load_feature_schema()
        threshold_cfg = load_threshold_config()

        return MetadataResponse(
            model_version=get_model_version(),
            threshold_policy=threshold_cfg["policy_name"],
            threshold_used=float(threshold_cfg["threshold"]),
            raw_input_features=feature_schema["raw_input_features"],
            engineered_features=feature_schema["engineered_features"],
            model_features=feature_schema["model_features"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metadata load failure: {e}")

# Endpoint 3: /predict 
@app.post("/predict", response_model=PredictResponse, response_model_exclude_none=True)
def predict(request: PredictRequest): # (αν λείπει field, αν έχει λάθος type, αν έχει extra field, αν το Time ή Amount είναι αρνητικό το FastAPI/Pydantic θα επιστρέψει 422 πριν καν μπει στη function logic. )
    try: # protective wrapper
        payload = _to_dict(request)
        result = _score_payload(payload) # shared scoring helper
        return PredictResponse(**result) # Το **result κάνει dictionary unpacking.
    except HTTPException: # Αν μέσα στο flow έχει ήδη προκύψει HTTPException, δεν θέλεις να την “τυλίξεις” ξανά ως generic 500
        raise
    except Exception as e: # Οτιδήποτε άλλο unexpected συμβεί, μετατρέπεται σε HTTP 500.
        raise HTTPException(status_code=500, detail=f"Prediction failure: {e}")

# Endpoint 4: /predict_by_id (Αυτό είναι το demo endpoint.Παίρνει μόνο row_id και κάνει lookup στο frozen demo CSV.)
@app.post("/predict_by_id", response_model=PredictResponse, response_model_exclude_none=True)
def predict_by_id(request: PredictByIdRequest):
    try:
        demo_df = load_demo_test_df() # cached DataFrame από το test_with_row_id.csv
        feature_schema = load_feature_schema() # για να ξέρεις ποια είναι η target column και ποια τα raw input features, ώστε να κάνεις σωστό lookup και να φτιάξεις το payload μόνο από τα raw input features.
        target_column = feature_schema["target_column"] # από το frozen schema παίρνεις ποια είναι η target column, για να την προσθέσεις στο output και να έχεις το true label μαζί με το prediction result.
        raw_input_features = feature_schema["raw_input_features"] # canonical raw feature list. Θα τη χρησιμοποιήσεις για να ξαναχτίσεις το payload από το demo row.
        # Φιλτράρισμα γραμμής με row_id
        matched = demo_df.loc[demo_df["row_id"] == request.row_id]
        if matched.empty: # Αν δεν βρέθηκε καμία γραμμή, επιστρέφεις 404.
            raise HTTPException(status_code=404, detail=f"row_id {request.row_id} not found")
        # Αφού ξέρεις ότι δεν είναι άδειο, παίρνεις την πρώτη γραμμή.Το row εδώ είναι Series. 
        row = matched.iloc[0]

        # Rebuild payload from raw canonical inputs only,
        # so demo scoring goes through the same serving pipeline as /predict
        payload = {col: float(row[col]) for col in raw_input_features}
        result = _score_payload(payload)

        result["row_id"] = int(request.row_id) # εμπλουτίζεις το result dict με το row_id για demo context
        result["true_label"] = int(row[target_column]) # μόνο για demo/testing/explainability. Δεν θα το είχες σε πραγματικό production inference endpoint.
        # περνάς το dict μέσα από το Pydantic response schema
        return PredictResponse(**result)
    # Error handling
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"predict_by_id failure: {e}")
    
    """/health
Απλό liveness check

/metadata
Επιστρέφει serving metadata από frozen configs

/predict
Validate request με PredictRequest
Convert σε dict
Preprocess + align
Predict probability
Apply threshold
Return JSON

/predict_by_id
Validate row_id
Find row στο frozen demo CSV
Rebuild raw payload
Πέρνα από το ίδιο scoring pipeline
Πρόσθεσε true_label
Return JSON"""
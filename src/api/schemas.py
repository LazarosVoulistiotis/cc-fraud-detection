"""schemas.py

Ορίζει αυστηρά:
- τι δέχεται το API
- τι επιστρέφει το API
- το contract για inference και metadata/versioning
- καθαρό και επαγγελματικό /docs output
"""

from typing import List, Literal, Optional
# Literal Χρησιμοποιείται όταν θέλεις ένα πεδίο να επιτρέπει μόνο συγκεκριμένες τιμές. 
# Optional Χρησιμοποιείται όταν ένα πεδίο μπορεί να υπάρχει ή να μην υπάρχει.

from pydantic import BaseModel, ConfigDict, Field
# BaseModel: Βασική κλάση για τη δημιουργία μοντέλων δεδομένων με Pydantic. Παρέχει αυτόματη επικύρωση και μετατροπή τύπων.
# Field: Χρησιμοποιείται για να ορίσει επιπλέον μεταδεδομένα για τα πεδία ενός Pydantic μοντέλου, όπως περιγραφές, περιορισμούς π.χ. ge=0 και προεπιλεγμένες τιμές.

# ξεκινάει το schema για το request του endpoint /predict
class PredictRequest(BaseModel):
    """
    Request schema for POST /predict.
    Accepts only the canonical raw input features expected by the serving pipeline.
    """
    model_config = ConfigDict(extra="forbid") # αν ο client στείλει επιπλέον fields που δεν τα περιμένω, απόρριψέ τα

    Time: float = Field(..., ge=0, description="Transaction time in seconds from dataset start.") # ... σημαίνει ότι το πεδίο είναι υποχρεωτικό, ge=0 σημαίνει ότι η τιμή πρέπει να είναι μεγαλύτερη ή ίση με 0, description εμφανίζεται στο auto-generated documentation του FastAPI
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0, description="Transaction amount.")
    # αν ο client στείλει επιπλέον fields που δεν τα περιμένω, απόρριψέ τα


# ξεκινάει το schema για το request του endpoint /predict_by_id
class PredictByIdRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_id: int = Field(..., ge=0, description="Frozen demo row_id from test_with_row_id.csv")


# ξεκινάει το schema για το response και των δύο endpoints /predict και /predict_by_id
class PredictResponse(BaseModel):
     model_version: str = Field(
        ...,
        description="Model artifact version used for inference.",
    )
     threshold_policy: str = Field(
        ...,
        description="Thresholding policy used to convert probability into label.",
    )
     threshold_used: float = Field(
        ...,
        description="Numeric threshold applied during prediction.",
    )
     fraud_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted fraud probability for the positive class.",
    )
     predicted_label: Literal["fraud", "legit"] = Field(
        ...,
        description="Final business label after thresholding.",
    )

    # Present only in /predict_by_id
     row_id: int | None = Field(
        default=None,
        description="Frozen demo row identifier.",
    )
     true_label: int | None = Field(
        default=None,
        description="Ground-truth label from the frozen demo dataset.",
    )


# ξεκινάει το schema για το response του endpoint /metadata
class MetadataResponse(BaseModel):
    """
    Response schema for GET /metadata.

    Acts as a mini model-registry / serving-metadata view for the deployed API.
    """
    model_version: str = Field(
        ...,
        description="Model artifact version currently served.",
    )
    model_artifact_path: str = Field(
        ...,
        description="Resolved path of the model artifact currently loaded by the API.",
    )
    git_commit: str = Field(
        ...,
        description="Git commit hash associated with the frozen model/build metadata.",
    )
    train_date: str = Field(
        ...,
        description="Training or freeze date recorded for the served model.",
    )

    threshold_policy: str = Field(
        ...,
        description="Thresholding policy currently active.",
    )
    threshold_policy_version: str = Field(
        ...,
        description="Version identifier of the active threshold policy.",
    )
    threshold_used: float = Field(
        ...,
        description="Threshold currently used by the API.",
    )

    schema_version: str = Field(
        ...,
        description="Version identifier of the frozen feature schema.",
    )

    raw_input_features: list[str] = Field(
        ...,
        description="Raw input fields expected by /predict.",
    )
    engineered_features: list[str] = Field(
        ...,
        description="Derived features created during preprocessing.",
    )
    model_features: list[str] = Field(
        ...,
        description="Final ordered feature list passed to the model.",
    )

    training_data_reference: str | None = Field(
        default=None,
        description="Reference path or identifier for the dataset used during training.",
    )
    training_target: str | None = Field(
        default=None,
        description="Target column used during model training.",
    )
    framework: str | None = Field(
        default=None,
        description="ML framework used to train the served model.",
    )
    task: str | None = Field(
        default=None,
        description="ML task type, e.g. binary_classification.",
    )
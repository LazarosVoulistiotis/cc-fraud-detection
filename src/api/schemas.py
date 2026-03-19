"""Το schemas.py:
    ορίζει αυστηρά τι δέχεται το API
    ορίζει αυστηρά τι επιστρέφει το API
    προστατεύει από λάθος inputs
    κάνει το /docs καθαρό και επαγγελματικό
    κρατά ενιαίο contract για real mode (/predict) και demo mode (/predict_by_id)"""

from typing import List, Literal, Optional
# Literal Χρησιμοποιείται όταν θέλεις ένα πεδίο να επιτρέπει μόνο συγκεκριμένες τιμές. 
# Optional Χρησιμοποιείται όταν ένα πεδίο μπορεί να υπάρχει ή να μην υπάρχει.

from pydantic import BaseModel, Field
# BaseModel: Βασική κλάση για τη δημιουργία μοντέλων δεδομένων με Pydantic. Παρέχει αυτόματη επικύρωση και μετατροπή τύπων.
# Field: Χρησιμοποιείται για να ορίσει επιπλέον μεταδεδομένα για τα πεδία ενός Pydantic μοντέλου, όπως περιγραφές, περιορισμούς π.χ. ge=0 και προεπιλεγμένες τιμές.

# ξεκινάει το schema για το request του endpoint /predict
class PredictRequest(BaseModel):
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
    class Config:
        extra = "forbid"

# ξεκινάει το schema για το request του endpoint /predict_by_id
class PredictByIdRequest(BaseModel):
    row_id: int = Field(..., ge=0, description="Frozen demo row_id from test_with_row_id.csv")

    class Config:
        extra = "forbid"

# ξεκινάει το schema για το response και των δύο endpoints /predict και /predict_by_id
class PredictResponse(BaseModel):
    model_version: str # ποιο model artifact χρησιμοποιήθηκε για την πρόβλεψη
    threshold_policy: str # API δεν απαντά μόνο με prediction, αλλά λέει και με ποια business rule το αποφάσισε.
    threshold_used: float # Το numeric threshold που χρησιμοποιήθηκε για τη μετατροπή probability → label.
    fraud_probability: float # πιθανότητα που επιστρέφει το model για την positive class
    predicted_label: Literal["fraud", "legit"] # η τιμή πρέπει να είναι μόνο "fraud" ή "legit", ανάλογα με το αν η fraud_probability είναι πάνω ή κάτω από το threshold_used

    # Only for /predict_by_id
    row_id: Optional[int] = None
    true_label: Optional[int] = None

# ξεκινάει το schema για το response του endpoint /metadata
class MetadataResponse(BaseModel):
    model_version: str
    threshold_policy: str
    threshold_used: float
    raw_input_features: List[str]
    engineered_features: List[str]
    model_features: List[str]
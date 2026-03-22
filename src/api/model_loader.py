"""model_loader.py

Resource manager του FastAPI app.

Υπεύθυνο για:
- εντοπισμό artifacts / configs
- ασφαλές loading
- καθαρά failures όταν λείπει κάτι
- caching των loaded resources
- παροχή έτοιμων objects στο υπόλοιπο API χωρίς επαναλαμβανόμενο loading
"""

from functools import lru_cache
"""Εδώ εισάγουμε το decorator lru_cache.
Αυτό είναι από τα πιο σημαντικά στοιχεία του αρχείου.
Όταν γράφουμε:@lru_cache(maxsize=1) πάνω από μία function, σημαίνει:
    την πρώτη φορά που καλείται η function, εκτελείται κανονικά
    το αποτέλεσμα αποθηκεύεται στη μνήμη
    την επόμενη φορά, επιστρέφεται το αποθηκευμένο αποτέλεσμα
    δεν ξανατρέχει ο κώδικας της function
Το maxsize=1 σημαίνει ότι κρατάμε μία cached εκδοχή του αποτελέσματος.
Για το δικό μας use case αυτό είναι ιδανικό, γιατί:
    έχουμε ένα threshold config
    ένα feature schema
    ένα model
    ένα demo dataframe
    Δεν χρειαζόμαστε πολλές εκδοχές"""

from pathlib import Path # για ασφαλή/καθαρό χειρισμό paths
from typing import Any, Dict # typing helpers
import json

import joblib # για serialization/deserialization μοντέλων, με joblib.load(...) φορτώνεις το trained model object πίσω στη μνήμη
import pandas as pd

# Αυτή η function επιστρέφει το root directory του project.
def get_project_root() -> Path:
    """
    Resolve project root directory from src/api/model_loader.py.

    src/api/model_loader.py -> parents[2] = project root
    parents[0] = api
    parents[1] = src
    parents[2] = project root
    """
    return Path(__file__).resolve().parents[2]


def validate_threshold_config(cfg: Dict[str, Any]) -> None:
    """
    Validate threshold serving config.

    Required keys:
    - policy_name
    - policy_version
    - threshold
    - reference_model_artifact

    Validation goals:
    - prevent silent schema drift
    - ensure threshold is numeric and in [0, 1]
    - ensure artifact path exists as a non-empty string reference
    """
    required_keys = [
        "policy_name",
        "policy_version",
        "threshold",
        "reference_model_artifact",
    ]

    missing = [key for key in required_keys if key not in cfg]
    if missing:
        raise ValueError(f"Threshold config missing required keys: {missing}")

    threshold = cfg["threshold"]
    if not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be numeric.")
    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError("threshold must be between 0 and 1.")

    artifact_ref = cfg["reference_model_artifact"]
    if not isinstance(artifact_ref, str) or not artifact_ref.strip():
        raise ValueError("reference_model_artifact must be a non-empty string.")


# Αυτή η function φορτώνει το threshold.json και επιστρέφει το περιεχόμενό του ως Python dict.
@lru_cache(maxsize=1)
def load_threshold_config() -> Dict[str, Any]:
    path = get_project_root() / "configs" / "threshold.json"
    if not path.exists():
        raise FileNotFoundError(f"Threshold config not found: {path}")
    # Local import για να αποφύγουμε circular imports αν χρειαστεί
    import json
    with open(path, "r", encoding="utf-8") as f: # ανοίγουμε το αρχείο σε read mode ("r") με utf-8 encoding. Το with εξασφαλίζει ότι το αρχείο θα κλείσει σωστά μετά το διάβασμα.
        return json.load(f)

# Αυτή η function φορτώνει το feature_schema.json και επιστρέφει το περιεχόμενό του ως Python dict.Και εδώ το αποτέλεσμα γίνεται cached.
@lru_cache(maxsize=1)
def load_feature_schema() -> Dict[str, Any]:
    path = get_project_root() / "configs" / "feature_schema.json"
    if not path.exists():
        raise FileNotFoundError(f"Feature schema not found: {path}")

    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Αυτή η function φορτώνει το model_metadata.json και επιστρέφει το περιεχόμενό του ως Python dict. Και εδώ το αποτέλεσμα γίνεται cached.
@lru_cache(maxsize=1)
def load_model_metadata() -> Dict[str, Any]:
    path = get_project_root() / "configs" / "model_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Model metadata file not found: {path}")

    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Αυτή η function επιστρέφει το path του μοντέλου που θα φορτωθεί. Το path αυτό καθορίζεται στο threshold config, με default fallback σε "models/xgb_final.joblib". Επίσης γίνεται cached για να μην ξαναδιαβάζει το config κάθε φορά.
@lru_cache(maxsize=1)
def get_model_path() -> Path:
    threshold_cfg = load_threshold_config() # φορτώνουμε το threshold config (με caching)
    rel_model_path = threshold_cfg.get("reference_model_artifact", "models/xgb_final.joblib") # αν στο JSON υπάρχει το key reference_model_artifact, πάρε αυτή την τιμή, αλλιώς χρησιμοποίησε το default path "models/xgb_final.joblib" 
    path = get_project_root() / rel_model_path # συνδυάζουμε το project root με το σχετικό path για να πάρουμε το απόλυτο path του μοντέλου

    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    return path

# Αυτή η function φορτώνει το μοντέλο από το path που επιστρέφει η get_model_path() και επιστρέφει το loaded model object. Και εδώ χρησιμοποιούμε caching για να μην ξαναφορτώνουμε το μοντέλο κάθε φορά.
@lru_cache(maxsize=1)
def load_model():
    model_path = get_model_path() # Καλείς την προηγούμενη function για να βρεις πού είναι το artifact.
    return joblib.load(model_path)


# Αυτή η function φορτώνει το demo test dataframe από ένα συγκεκριμένο CSV αρχείο και επιστρέφει το DataFrame. 
def load_demo_test_df() -> pd.DataFrame:
    path = get_project_root() / "data" / "data_interim" / "splits_week8" / "test_with_row_id.csv" # το path του demo CSV αρχείου που θέλουμε να φορτώσουμε
    if not path.exists():
        raise FileNotFoundError(f"Demo test CSV not found: {path}")

    df = pd.read_csv(path)
    if "row_id" not in df.columns: # ελέγχουμε αν υπάρχει η στήλη 'row_id' στο DataFrame, γιατί είναι απαραίτητη για το demo. Αν δεν υπάρχει, σηκώνουμε ένα ValueError με κατάλληλο μήνυμα.
        raise ValueError(f"'row_id' column missing in demo CSV: {path}")

    return df

# Αυτή η function επιστρέφει ένα string με το όνομα/version του model.
def get_model_version() -> str:
    return get_model_path().stem
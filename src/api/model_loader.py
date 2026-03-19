"""Το model_loader.py είναι υπεύθυνο για να:
    εντοπίζει σωστά τα artifacts/configs
    τα φορτώνει με ασφάλεια
    αποτυγχάνει καθαρά όταν λείπει κάτι
    κρατάει loaded resources στη μνήμη
    δίνει στο υπόλοιπο API έτοιμα objects χωρίς επαναλαμβανόμενο loading
Με πολύ απλά λόγια είναι το “resource manager” του FastAPI app μας."""

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

import joblib # για serialization/deserialization μοντέλων, με joblib.load(...) φορτώνεις το trained model object πίσω στη μνήμη
import pandas as pd

# Αυτή η function επιστρέφει το root directory του project.
def get_project_root() -> Path:
    """
    src/api/model_loader.py -> parents[2] = project root 
    (parents[0] είναι ο φάκελος api)
    __file__ Είναι το path του τρέχοντος αρχείου, δηλαδή του model_loader.py.
    πολύ σημαντικό στο deployment, γιατί δεν θέλουμε relative path bugs
    """
    return Path(__file__).resolve().parents[2]

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

# Αυτή η function φορτώνει το demo test dataframe από ένα συγκεκριμένο CSV αρχείο και επιστρέφει το DataFrame. Επίσης γίνεται cached για να μην ξαναφορτώνουμε το CSV κάθε φορά.
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
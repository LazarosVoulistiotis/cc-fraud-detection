"""Το preprocess.py:
    μετατρέπει raw API input σε model-ready input
    αναπαράγει τη feature engineering λογική του training
    προστατεύει από missing required inputs
    επιβάλλει την ακριβή σειρά features
    επιστρέφει deterministic one-row DataFrame για inference"""

from typing import Any, Dict

import numpy as np
import pandas as pd

# Αυτή η function: παίρνει ένα DataFrame, προσθέτει τα engineered columns, επιστρέφει νέο DataFrame
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the engineered features expected by the frozen Week 17 model:
    - Hour
    - hour_sin
    - hour_cos
    - Amount_log1p
    """
    required = {"Time", "Amount"} # set με τις στήλες που είναι απαραίτητες για να γίνει feature engineering
    missing = required - set(df.columns) # Ό,τι μείνει, είναι οι στήλες που λείπουν
    if missing:
        raise ValueError(f"Missing columns for feature engineering: {sorted(missing)}")
    # Αντί να αλλάζεις το αρχικό df in-place, φτιάχνεις αντίγραφο. Έτσι αποφεύγεις side effects που μπορεί να μπερδέψουν αν χρησιμοποιήσεις το ίδιο df για πολλαπλές προετοιμασίες.
    out = df.copy()

    # Convert dataset-relative seconds to hour-of-day bucket [0, 23]
    out["Hour"] = np.floor(out["Time"] / 3600.0).astype(int) % 24 

    # Cyclical encoding (transforms periodic features (e.g., hours, months) into two numerical features—sine and cosine—to preserve their continuous, looping nature. By mapping data onto a circle, it ensures, for example, that December (12) is recognized as close to January (1) rather than distant, preventing artificial discontinuities, particularly in distance-based models and neural networks)
    out["hour_sin"] = np.sin(2 * np.pi * out["Hour"] / 24.0) # μετατρέπει την ώρα σε γωνία πάνω σε κύκλο
    out["hour_cos"] = np.cos(2 * np.pi * out["Hour"] / 24.0) # δίνει τη sine συνιστώσα αυτής της γωνίας

    # Log-transform amount (αν οποιοδήποτε ποσό είναι αρνητικό, σήκωσε error)
    if (out["Amount"] < 0).any():
        raise ValueError("Amount must be non-negative for Amount_log1p.")
    out["Amount_log1p"] = np.log1p(out["Amount"]) # log1p(x) = log(1 + x), αποφεύγει προβλήματα με το log(0) και διατηρεί την τάξη μεγέθους των ποσών

    return out

# Αυτή η function: παίρνει ένα DataFrame, το ευθυγραμμίζει ακριβώς με τη σειρά των features που περίμενε το model όταν εκπαιδεύτηκε, επιστρέφει νέο DataFrame
def align_features(df: pd.DataFrame, feature_schema: Dict[str, Any]) -> pd.DataFrame:
    """
    Reorder columns exactly as frozen in feature_schema['model_features'].
    Extra columns are ignored. Missing columns raise an error.
    """
    # Πάρε το frozen feature order από το feature_schema
    model_features = feature_schema["model_features"]
    # Περνάς μία-μία τις στήλες του frozen schema και ελέγχεις αν υπάρχουν στο DataFrame.Όσες δεν υπάρχουν, μπαίνουν στη λίστα missing. 
    missing = [col for col in model_features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing features after preprocessing/alignment: {missing}")
    # exact reordering: κρατάς μόνο τις στήλες που είναι στο model_features και τις βάζεις με τη σειρά που είναι εκεί. Αν υπάρχουν επιπλέον στήλες στο df που δεν είναι στο model_features, απλά τις αγνοείς.
    aligned = df.loc[:, model_features].copy()
    return aligned

# “orchestrator” function
def prepare_single_payload(payload_dict: Dict[str, Any], feature_schema: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert one incoming JSON payload into a one-row model-ready DataFrame.
    Steps:
    1. Validate raw input keys
    2. Add engineered features
    3. Align to exact frozen feature order
    """
    raw_input_features = feature_schema["raw_input_features"]

    missing = [col for col in raw_input_features if col not in payload_dict]
    if missing:
        raise ValueError(f"Missing raw input fields: {missing}")

    # Keep only canonical raw inputs in the frozen order
    row = {col: payload_dict[col] for col in raw_input_features}
    df = pd.DataFrame([row]) # φτιάχνει ένα DataFrame με μία μόνο γραμμή, όπου τα columns είναι τα raw_input_features και οι τιμές τους είναι αυτές που έδωσε ο client στο payload_dict
    # Καλείς την πρώτη function. Αυτή επιστρέφει νέο DataFrame με τα engineered features προστιθέμενα.
    df = add_engineered_features(df)
    # Καλείς τη δεύτερη function. Αυτή επιστρέφει νέο DataFrame με ακριβώς τις στήλες που περίμενε το model, στην ακριβή σειρά που τις περίμενε.
    df = align_features(df, feature_schema)
    # Επιστρέφεις το τελικό one-row aligned DataFrame. Αυτό είναι το τελικό input που θα μπει στο XGBoost model.
    # Force numeric float dtype for model serving
    df = df.astype(float)

    return df
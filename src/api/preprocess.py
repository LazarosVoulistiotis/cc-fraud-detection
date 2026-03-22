"""preprocess.py

Μετατρέπει raw API input σε model-ready input.
Αναπαράγει τη feature engineering λογική του training.
Προστατεύει από missing / unexpected / invalid inputs.
Επιβάλλει την ακριβή σειρά features του frozen serving schema.
Επιστρέφει deterministic one-row DataFrame για inference.
"""

from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

# Σταθερές
REQUIRED_SCHEMA_KEYS = ("raw_input_features", "engineered_features", "model_features")
ENGINEERING_SOURCE_COLUMNS = ("Time", "Amount")

# Helper function. Το _ στην αρχή του ονόματος σημαίνει: “ιδιωτική / internal χρήση” και δεν προορίζεται ως public API του module
def _ensure_list_of_strings(name: str, value: Any) -> list[str]:
    """Validate that a schema field is a list of strings."""
    if not isinstance(value, list):
        raise ValueError(f"feature_schema['{name}'] must be a list.")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"feature_schema['{name}'] must contain only strings.")
    return value

# Αυτή η helper function εντοπίζει duplicates.
def _find_duplicates(items: Iterable[str]) -> list[str]:
    """Return duplicate values while preserving first duplicate discovery order."""
    seen = set() # Για να παρακολουθούμε ποια items έχουμε ήδη δει. Το set είναι πολύ γρήγορο για membership checks
    duplicates = []
    for item in items:
        if item in seen and item not in duplicates:
            duplicates.append(item)
        seen.add(item)
    return duplicates

# Ελέγχει ότι το frozen schema είναι έγκυρο
def validate_feature_schema(feature_schema: Dict[str, Any]) -> None:
    """
    Validate the frozen feature schema used by the serving pipeline.

    Checks:
    - required keys exist
    - values are lists of strings
    - no duplicate feature names
    - model_features match raw_input_features + engineered_features exactly
    """
    if not isinstance(feature_schema, dict):
        raise ValueError("feature_schema must be a dictionary.")

    missing_keys = [key for key in REQUIRED_SCHEMA_KEYS if key not in feature_schema]
    if missing_keys:
        raise ValueError(f"feature_schema missing required keys: {missing_keys}")
   
    # να διασφαλίσει ότι και τα 3 schema fields είναι λίστες από strings
    raw_input_features = _ensure_list_of_strings(
        "raw_input_features", feature_schema["raw_input_features"]
    )
    engineered_features = _ensure_list_of_strings(
        "engineered_features", feature_schema["engineered_features"]
    )
    model_features = _ensure_list_of_strings(
        "model_features", feature_schema["model_features"]
    )

    for name, values in (
        ("raw_input_features", raw_input_features),
        ("engineered_features", engineered_features),
        ("model_features", model_features),
    ):
        duplicates = _find_duplicates(values)
        if duplicates:
            raise ValueError(f"Duplicate feature names found in {name}: {duplicates}")
   
    # Έλεγχος ακριβούς αντιστοίχισης model_features
    expected_model_features = raw_input_features + engineered_features
    if model_features != expected_model_features:
        raise ValueError(
            "Frozen schema mismatch: "
            "model_features must equal raw_input_features + engineered_features "
            "in exact order."
        )

# Αυτή η helper function μετατρέπει όλες τις στήλες σε numeric και απορρίπτει μη έγκυρες τιμές.
def _coerce_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all columns to numeric and reject NaN / inf values.
    """
    out = df.apply(pd.to_numeric, errors="raise")

    # Μετατρέπει το DataFrame σε NumPy array από floats για να γίνει numeric/global check εύκολα με NumPy.
    values = out.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Input contains NaN or infinite values.")

    return out


# Αυτή η συνάρτηση προσθέτει τα engineered features.
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the engineered features expected by the frozen model:
    - Hour
    - hour_sin
    - hour_cos
    - Amount_log1p
    """
    missing = sorted(set(ENGINEERING_SOURCE_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns for feature engineering: {missing}")

    out = _coerce_numeric_dataframe(df.copy())

    # Business rule: Time non-negative and Amount non-negative
    if (out["Time"] < 0).any():
        raise ValueError("Time must be non-negative.")
    if (out["Amount"] < 0).any():
        raise ValueError("Amount must be non-negative for Amount_log1p.")

    # Convert dataset-relative seconds to hour-of-day bucket [0, 23]
    out["Hour"] = np.floor(out["Time"] / 3600.0).astype(int) % 24

    # Cyclical encoding for hour-of-day
    out["hour_sin"] = np.sin(2.0 * np.pi * out["Hour"] / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * out["Hour"] / 24.0)

    # Log-transform amount
    out["Amount_log1p"] = np.log1p(out["Amount"])

    return out


# Αυτή η συνάρτηση φροντίζει ώστε το τελικό DataFrame να έχει ακριβώς τη σωστή σειρά στηλών.
def align_features(df: pd.DataFrame, feature_schema: Dict[str, Any]) -> pd.DataFrame:
    """
    Reorder columns exactly as frozen in feature_schema['model_features'].
    Extra columns are ignored.
    Missing columns raise an error.
    """
    validate_feature_schema(feature_schema)

    model_features = feature_schema["model_features"]
    missing = [col for col in model_features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing features after preprocessing/alignment: {missing}")

    aligned = df.loc[:, model_features].copy()
    aligned = _coerce_numeric_dataframe(aligned)

    return aligned.astype(float)


# Αυτή είναι η κύρια public function του module. Είναι η function που πιθανότατα θα καλεί το API endpoint πριν το inference.
def prepare_single_payload(payload_dict: Dict[str, Any], feature_schema: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert one incoming JSON payload into a one-row model-ready DataFrame.

    Steps:
    1. Validate frozen feature schema
    2. Validate raw input keys
    3. Build one-row DataFrame in canonical raw order
    4. Add engineered features
    5. Align to exact frozen model feature order
    6. Return float DataFrame ready for inference
    """
    if not isinstance(payload_dict, dict):
        raise ValueError("payload_dict must be a dictionary.")

    validate_feature_schema(feature_schema)

    raw_input_features = feature_schema["raw_input_features"]

    missing = [col for col in raw_input_features if col not in payload_dict]
    if missing:
        raise ValueError(f"Missing raw input fields: {missing}")

    unexpected = sorted(set(payload_dict.keys()) - set(raw_input_features))
    if unexpected:
        raise ValueError(f"Unexpected raw input fields: {unexpected}")

    # Keep only canonical raw inputs in the frozen order
    row = {col: payload_dict[col] for col in raw_input_features}
    df = pd.DataFrame([row], columns=raw_input_features)

    # Normalize raw inputs to numeric early, so failures are explicit
    df = _coerce_numeric_dataframe(df)

    # Add engineered features and align to model schema
    df = add_engineered_features(df)
    df = align_features(df, feature_schema)

    return df
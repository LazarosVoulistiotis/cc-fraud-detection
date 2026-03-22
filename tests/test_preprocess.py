# test_preprocess.py

# για να υπολογίσουμε τις αναμενόμενες μαθηματικές τιμές στα tests και να τις συγκρίνουμε με αυτά που παράγει ο κώδικας
import math

import numpy as np
import pandas as pd
import pytest

# functions που θα τεστάρουμε
from src.api.preprocess import (
    add_engineered_features,
    align_features,
    prepare_single_payload,
    validate_feature_schema,
)


# Αυτό το decorator λέει στο pytest: αυτή η function δεν είναι test, είναι helper provider δεδομένων και μπορεί να τη χρησιμοποιούν πολλά tests
@pytest.fixture
def feature_schema():
    return {
        "raw_input_features": ["Time", "V1", "V2", "Amount"],
        "engineered_features": ["Hour", "hour_sin", "hour_cos", "Amount_log1p"],
        "model_features": [
            "Time",
            "V1",
            "V2",
            "Amount",
            "Hour",
            "hour_sin",
            "hour_cos",
            "Amount_log1p",
        ],
    }


@pytest.fixture
def valid_payload():
    return {
        "Time": 7200.0,   # 7200 δευτερόλεπτα, δηλαδή 2 ώρες from dataset start
        "V1": 1.5,
        "V2": -0.25,
        "Amount": 99.0,
    }


# Εδώ ξεκινούν τα actual tests. 
# Test: valid schema accepted
def test_validate_feature_schema_accepts_valid_schema(feature_schema):
    validate_feature_schema(feature_schema)


# Test: valid schema accepted
def test_validate_feature_schema_rejects_duplicate_names():
    bad_schema = {
        "raw_input_features": ["Time", "V1", "V1", "Amount"],
        "engineered_features": ["Hour", "hour_sin", "hour_cos", "Amount_log1p"],
        "model_features": [
            "Time",
            "V1",
            "V1",
            "Amount",
            "Hour",
            "hour_sin",
            "hour_cos",
            "Amount_log1p",
        ],
    }

    with pytest.raises(ValueError, match="Duplicate feature names"):
        validate_feature_schema(bad_schema)


# Test: model feature order mismatch rejected
def test_validate_feature_schema_rejects_model_feature_order_mismatch():
    bad_schema = {
        "raw_input_features": ["Time", "V1", "V2", "Amount"],
        "engineered_features": ["Hour", "hour_sin", "hour_cos", "Amount_log1p"],
        "model_features": [  # intentionally wrong order
            "Time",
            "V2",
            "V1",
            "Amount",
            "Hour",
            "hour_sin",
            "hour_cos",
            "Amount_log1p",
        ],
    }

    with pytest.raises(ValueError, match="Frozen schema mismatch"):
        validate_feature_schema(bad_schema)


# Test: engineered features are created correctly
def test_add_engineered_features_creates_expected_columns():
    df = pd.DataFrame(
        [{"Time": 7200.0, "V1": 1.5, "V2": -0.25, "Amount": 99.0}]
    )

    out = add_engineered_features(df)

    assert "Hour" in out.columns
    assert "hour_sin" in out.columns
    assert "hour_cos" in out.columns
    assert "Amount_log1p" in out.columns

    assert out.loc[0, "Hour"] == 2
    assert np.isclose(out.loc[0, "hour_sin"], math.sin(2 * math.pi * 2 / 24.0))
    assert np.isclose(out.loc[0, "hour_cos"], math.cos(2 * math.pi * 2 / 24.0))
    assert np.isclose(out.loc[0, "Amount_log1p"], math.log1p(99.0))


# Test: engineered features are created correctly
def test_add_engineered_features_rejects_missing_required_columns():
    df = pd.DataFrame([{"Time": 100.0, "V1": 1.0}])

    with pytest.raises(ValueError, match="Missing columns for feature engineering"):
        add_engineered_features(df)


# Test: missing required columns rejected
def test_add_engineered_features_rejects_negative_amount():
    df = pd.DataFrame(
        [{"Time": 100.0, "V1": 1.0, "V2": 2.0, "Amount": -5.0}]
    )

    with pytest.raises(ValueError, match="Amount must be non-negative"):
        add_engineered_features(df)


# Test: negative time rejected
def test_add_engineered_features_rejects_negative_time():
    df = pd.DataFrame(
        [{"Time": -1.0, "V1": 1.0, "V2": 2.0, "Amount": 10.0}]
    )

    with pytest.raises(ValueError, match="Time must be non-negative"):
        add_engineered_features(df)


# Test: negative time rejected
def test_align_features_reorders_columns_exactly(feature_schema):
    df = pd.DataFrame(
        [
            {
                "hour_cos": 0.5,
                "V2": -0.25,
                "Amount": 99.0,
                "Hour": 2,
                "Time": 7200.0,
                "Amount_log1p": math.log1p(99.0),
                "V1": 1.5,
                "hour_sin": 0.8660254,
                "extra_column": 123.0,
            }
        ]
    )

    aligned = align_features(df, feature_schema)

    assert list(aligned.columns) == feature_schema["model_features"]
    assert "extra_column" not in aligned.columns


# Test: align features fails if required feature missing
def test_align_features_raises_if_required_feature_missing(feature_schema):
    df = pd.DataFrame(
        [
            {
                "Time": 7200.0,
                "V1": 1.5,
                "V2": -0.25,
                "Amount": 99.0,
                "Hour": 2,
                "hour_sin": 0.5,
                # missing hour_cos
                "Amount_log1p": math.log1p(99.0),
            }
        ]
    )

    with pytest.raises(ValueError, match="Missing features after preprocessing/alignment"):
        align_features(df, feature_schema)


# Test: prepare_single_payload returns aligned one-row float DataFrame
def test_prepare_single_payload_returns_one_row_aligned_float_dataframe(
    feature_schema, valid_payload
):
    df = prepare_single_payload(valid_payload, feature_schema)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, len(feature_schema["model_features"]))
    assert list(df.columns) == feature_schema["model_features"]

    # all output dtypes should be float
    assert all(dtype == float for dtype in df.dtypes)

    # sanity checks on engineered values
    assert df.loc[0, "Hour"] == 2.0
    assert np.isclose(df.loc[0, "Amount_log1p"], math.log1p(99.0))


# Test: deterministic output
def test_prepare_single_payload_is_deterministic(feature_schema, valid_payload):
    df1 = prepare_single_payload(valid_payload, feature_schema)
    df2 = prepare_single_payload(valid_payload, feature_schema)

    pd.testing.assert_frame_equal(df1, df2)


# Test: missing raw field rejected
def test_prepare_single_payload_rejects_missing_raw_field(feature_schema, valid_payload):
    bad_payload = dict(valid_payload)
    bad_payload.pop("V2")

    with pytest.raises(ValueError, match="Missing raw input fields"):
        prepare_single_payload(bad_payload, feature_schema)


# Test: unexpected raw field rejected
def test_prepare_single_payload_rejects_unexpected_raw_field(feature_schema, valid_payload):
    bad_payload = dict(valid_payload)
    bad_payload["unexpected_feature"] = 123

    with pytest.raises(ValueError, match="Unexpected raw input fields"):
        prepare_single_payload(bad_payload, feature_schema)


# Test: non-numeric input rejected
def test_prepare_single_payload_rejects_non_numeric_input(feature_schema, valid_payload):
    bad_payload = dict(valid_payload)
    bad_payload["V1"] = "not-a-number"

    with pytest.raises(ValueError):
        prepare_single_payload(bad_payload, feature_schema)


# Test: NaN rejected
def test_prepare_single_payload_rejects_nan_input(feature_schema, valid_payload):
    bad_payload = dict(valid_payload)
    bad_payload["V1"] = float("nan")

    with pytest.raises(ValueError, match="NaN or infinite values"):
        prepare_single_payload(bad_payload, feature_schema)


# Test: inf rejected
def test_prepare_single_payload_rejects_inf_input(feature_schema, valid_payload):
    bad_payload = dict(valid_payload)
    bad_payload["V1"] = float("inf")

    with pytest.raises(ValueError, match="NaN or infinite values"):
        prepare_single_payload(bad_payload, feature_schema)
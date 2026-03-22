from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.api.main as main
import src.api.model_loader as model_loader
from src.api.main import app


client = TestClient(app)


class FakeModel:
    """Small fake model for deterministic golden-path API testing."""

    def predict_proba(self, X):
        # Always return fixed fraud probability
        return np.array([[0.9, 0.1] for _ in range(len(X))], dtype=float)


@pytest.fixture(autouse=True)
def mock_golden_runtime(monkeypatch):
    """
    Make golden test independent from:
    - real model artifact
    - real demo CSV
    - real metadata file
    - real threshold config
    """
    fake_model_path = Path("models/xgb_final.joblib")

    fake_feature_schema = {
        "schema_version": "test-schema-v1",
        "target_column": "Class",
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

    fake_threshold_cfg = {
        "policy_name": "test_policy",
        "policy_version": "test_policy_v1",
        "threshold": 0.5,
        "reference_model_artifact": "models/xgb_final.joblib",
    }

    fake_demo_df = pd.DataFrame(
        [
            {
                "row_id": 1,
                "Time": 7200.0,
                "V1": 1.5,
                "V2": -0.25,
                "Amount": 99.0,
                "Class": 0,
            }
        ]
    )

    monkeypatch.setattr(main, "load_model", lambda: FakeModel())
    monkeypatch.setattr(main, "get_model_path", lambda: fake_model_path)
    monkeypatch.setattr(main, "get_model_version", lambda: "xgb_final")
    monkeypatch.setattr(main, "load_demo_test_df", lambda: fake_demo_df)
    monkeypatch.setattr(main, "load_feature_schema", lambda: fake_feature_schema)
    monkeypatch.setattr(main, "load_threshold_config", lambda: fake_threshold_cfg)

    monkeypatch.setattr(model_loader, "get_model_path", lambda: fake_model_path)
    monkeypatch.setattr(model_loader, "get_model_version", lambda: "xgb_final")


def test_predict_by_id_golden_case():
    response = client.post("/predict_by_id", json={"row_id": 1})

    assert response.status_code == 200, response.text
    assert "X-Request-ID" in response.headers

    body = response.json()

    assert body["row_id"] == 1
    assert body["true_label"] == 0
    assert body["model_version"] == "xgb_final"
    assert body["threshold_policy"] == "test_policy"
    assert body["threshold_used"] == 0.5

    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["predicted_label"] in {"fraud", "legit"}

    if body["fraud_probability"] >= body["threshold_used"]:
        assert body["predicted_label"] == "fraud"
    else:
        assert body["predicted_label"] == "legit"
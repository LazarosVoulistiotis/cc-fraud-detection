from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

KNOWN_ROW_ID = 1
EXPECTED_LABEL = "legit"


def test_predict_by_id_golden_case():
    response = client.post("/predict_by_id", json={"row_id": KNOWN_ROW_ID})

    assert response.status_code == 200, response.text

    body = response.json()
    assert body["row_id"] == KNOWN_ROW_ID
    assert body["predicted_label"] == EXPECTED_LABEL

    if EXPECTED_LABEL == "fraud":
        assert body["fraud_probability"] >= body["threshold_used"]
    else:
        assert body["fraud_probability"] < body["threshold_used"]
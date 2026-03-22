# tests/test_api.py

# Το TestClient είναι εργαλείο για να κάνεις δοκιμαστικά HTTP requests στο app σου χωρίς να χρειάζεται να τρέχει κανονικά server με uvicorn.
from fastapi.testclient import TestClient

from src.api.main import app


# Δημιουργία test client
client = TestClient(app)


# Helper function για να φτιάχνουμε έγκυρα payloads για το /predict endpoint.
def make_valid_payload() -> dict:
    return {
        "Time": 0.0, # το Time είναι σε seconds από την αρχή του dataset, δεν έχει πραγματική σημασία για το test μας, απλά θέλουμε να είναι float
        **{f"V{i}": 0.0 for i in range(1, 29)}, # τα V1-V28 είναι τα anonymized features, πάλι δεν έχει σημασία τι τιμές έχουν για το test μας, απλά θέλουμε να υπάρχουν και να είναι float
        "Amount": 0.0, # το Amount είναι το ποσό της συναλλαγής, πάλι δεν έχει σημασία για το test μας, απλά θέλουμε να υπάρχει και να είναι float
    }


# Test για το /health endpoint
def test_health_returns_200():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "X-Request-ID" in response.headers # συνδέεται με το middleware που φτιάξαμε στο main.py


# Test για το /metadata endpoint, ελέγχουμε ότι επιστρέφει όλα τα πεδία που έχουμε ορίσει στο API και ότι έχουν λογικές τιμές (π.χ. το model_version είναι μη κενό string, το threshold_used είναι μεταξύ 0 και 1 κλπ).
def test_metadata_returns_versioning_fields():
    response = client.get("/metadata")

    assert response.status_code == 200, response.text
    assert "X-Request-ID" in response.headers

    body = response.json()

    assert "model_version" in body
    assert isinstance(body["model_version"], str)
    assert body["model_version"] != ""

    assert "model_artifact_path" in body
    assert isinstance(body["model_artifact_path"], str)
    assert body["model_artifact_path"].endswith(".joblib")

    assert "git_commit" in body
    assert isinstance(body["git_commit"], str)
    assert body["git_commit"] != ""

    assert "train_date" in body
    assert isinstance(body["train_date"], str)
    assert body["train_date"] != ""

    assert "threshold_policy" in body
    assert isinstance(body["threshold_policy"], str)
    assert body["threshold_policy"] != ""

    assert "threshold_policy_version" in body
    assert isinstance(body["threshold_policy_version"], str)
    assert body["threshold_policy_version"] != ""

    assert "threshold_used" in body
    assert isinstance(body["threshold_used"], (int, float))
    assert 0.0 <= body["threshold_used"] <= 1.0

    assert "schema_version" in body
    assert isinstance(body["schema_version"], str)
    assert body["schema_version"] != ""

    assert "raw_input_features" in body
    assert isinstance(body["raw_input_features"], list)
    assert len(body["raw_input_features"]) > 0

    assert "engineered_features" in body
    assert isinstance(body["engineered_features"], list)
    assert len(body["engineered_features"]) > 0

    assert "model_features" in body
    assert isinstance(body["model_features"], list)
    assert len(body["model_features"]) == (
        len(body["raw_input_features"]) + len(body["engineered_features"])
    )

    assert "training_data_reference" in body
    assert body["training_data_reference"] is None or isinstance(body["training_data_reference"], str)

    assert "training_target" in body
    assert body["training_target"] is None or isinstance(body["training_target"], str)

    assert "framework" in body
    assert body["framework"] is None or isinstance(body["framework"], str)

    assert "task" in body
    assert body["task"] is None or isinstance(body["task"], str)


# Αυτό είναι το βασικό functional test για το prediction endpoint.
def test_predict_valid_payload_returns_probability_between_0_and_1():
    payload = make_valid_payload()

    response = client.post("/predict", json=payload)

    assert response.status_code == 200, response.text
    assert "X-Request-ID" in response.headers

    body = response.json()

    assert "fraud_probability" in body
    assert 0.0 <= body["fraud_probability"] <= 1.0 # Έλεγχος ότι η πιθανότητα είναι μεταξύ 0 και 1

    assert "predicted_label" in body
    assert body["predicted_label"] in {"fraud", "legit"}

    assert "threshold_used" in body
    assert isinstance(body["threshold_used"], (int, float))

    assert "threshold_policy" in body
    assert isinstance(body["threshold_policy"], str)

    assert "model_version" in body
    assert isinstance(body["model_version"], str)


# Test για να ελέγξουμε ότι αν λείπει κάποιο υποχρεωτικό πεδίο (π.χ. V28) τότε το API επιστρέφει 422 Unprocessable Entity.
def test_predict_missing_field_returns_422():
    payload = make_valid_payload()
    payload.pop("V28") # αφαιρούμε το V28 για να κάνουμε το payload άκυρο

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "X-Request-ID" in response.headers

    body = response.json()
    assert "detail" in body
    assert isinstance(body["detail"], list)
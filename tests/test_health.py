from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "X-Request-ID" in response.headers
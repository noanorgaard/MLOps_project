from fastapi.testclient import TestClient
from ai_vs_human.api import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200

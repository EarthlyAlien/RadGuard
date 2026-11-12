"""HTTP-layer tests: validation, size limits, health, OpenAPI."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import __version__
from app.limits import MAX_BODY_BYTES, MAX_TEXT_CHARS
from app.main import app


def test_health_is_ready(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["ready"] is True
    assert body["version"] == __version__
    assert "measurements" in body["validators"]


def test_validate_flags_readme_example(client: TestClient) -> None:
    response = client.post(
        "/validate",
        json={
            "findings_text": (
                "There is a 4 mm non-obstructing calculus in the left kidney. No hydronephrosis."
            ),
            "impression_text": "4 cm stone in the right kidney.",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["is_safe"] is False
    categories = {flag["category"] for flag in body["flags"]}
    assert "MEASUREMENT_MISMATCH" in categories
    assert "LATERALITY_CONFLICT" in categories
    assert body["latency_ms"] >= 0.0


def test_validate_safe_unit_conversion(client: TestClient) -> None:
    response = client.post(
        "/validate",
        json={
            "findings_text": "10 mm nodule. No pneumothorax.",
            "impression_text": "1 cm nodule. No pneumothorax.",
        },
    )
    assert response.status_code == 200
    assert response.json()["is_safe"] is True


def test_extra_fields_rejected(client: TestClient) -> None:
    response = client.post(
        "/validate",
        json={
            "findings_text": "10 mm nodule.",
            "impression_text": "1 cm nodule.",
            "extra": "nope",
        },
    )
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "VALIDATION_ERROR"


def test_null_bytes_rejected(client: TestClient) -> None:
    response = client.post(
        "/validate",
        json={"findings_text": "10 mm\x00 nodule.", "impression_text": "1 cm nodule."},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_field_too_long_rejected(client: TestClient) -> None:
    response = client.post(
        "/validate",
        json={
            "findings_text": "n" * (MAX_TEXT_CHARS + 1),
            "impression_text": "1 cm nodule.",
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_missing_fields_rejected(client: TestClient) -> None:
    response = client.post("/validate", json={"findings_text": "10 mm nodule."})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_payload_too_large() -> None:
    large_client = TestClient(app)
    response = large_client.post(
        "/validate",
        content=b"x" * (MAX_BODY_BYTES + 10),
        headers={"Content-Type": "application/json", "Content-Length": str(MAX_BODY_BYTES + 10)},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_openapi_documents_routes_and_models(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    assert spec["info"]["title"] == "RadGuard"
    assert "/validate" in spec["paths"]
    assert "/health" in spec["paths"]
    validate = spec["paths"]["/validate"]["post"]
    assert "422" in validate["responses"]
    assert "413" in validate["responses"]
    schemas = spec["components"]["schemas"]
    assert "ValidationRequest" in schemas
    assert "ValidationResponse" in schemas
    assert "HealthResponse" in schemas
    assert "ErrorResponse" in schemas
    flag_enum = schemas["FlagCategory"]["enum"]
    assert "MEASUREMENT_MISMATCH" in flag_enum
    assert "PARSE_AMBIGUITY" in flag_enum

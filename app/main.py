"""FastAPI entrypoint for the RadGuard validation microservice."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from app import __version__
from app.engine import evaluate
from app.errors import ErrorResponse, error_payload
from app.limits import MAX_BODY_BYTES
from app.schemas import HealthResponse, ValidationRequest, ValidationResponse

app = FastAPI(
    title="RadGuard",
    description=(
        "Deterministic, sub-10ms hallucination and contradiction guardrail "
        "for radiology findings/impression pairs. No external LLM calls on the "
        "validation path. Flags measurement/unit drift, laterality swaps, and "
        "negation polarity flips."
    ),
    version=__version__,
    contact={"name": "RadGuard", "url": "https://github.com/EarthlyAlien/RadGuard"},
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    openapi_tags=[
        {"name": "validation", "description": "Deterministic findings/impression checks."},
        {"name": "ops", "description": "Liveness and readiness probes."},
    ],
)


@app.middleware("http")
async def limit_request_body(request: Request, call_next: Any) -> Any:
    """Reject oversized bodies before JSON parsing."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            length = int(content_length)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content=error_payload("INVALID_CONTENT_LENGTH", "Content-Length is not an integer."),
            )
        if length > MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content=error_payload(
                    "PAYLOAD_TOO_LARGE",
                    f"Request body exceeds {MAX_BODY_BYTES} bytes.",
                ),
            )
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return structured 422 errors for malformed or oversized fields."""
    details = []
    for error in exc.errors():
        details.append(
            {
                "loc": list(error.get("loc", [])),
                "msg": error.get("msg"),
                "type": error.get("type"),
            }
        )
    return JSONResponse(
        status_code=422,
        content=error_payload("VALIDATION_ERROR", "Request validation failed.", details),
    )


@app.exception_handler(HTTPException)
async def http_error_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    """Normalize HTTPException into the error envelope."""
    message = exc.detail if isinstance(exc.detail, str) else "HTTP error"
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload("HTTP_ERROR", message),
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_request: Request, _exc: Exception) -> JSONResponse:
    """Never leak stack traces; fail closed at the HTTP boundary."""
    return JSONResponse(
        status_code=500,
        content=error_payload("INTERNAL_ERROR", "Unexpected server error."),
    )


def _readiness() -> bool:
    """Cheap self-check: equivalent 10 mm / 1 cm pair must stay safe."""
    try:
        result = evaluate(
            "There is a 10 mm nodule. No pneumothorax.",
            "1 cm nodule. No pneumothorax.",
        )
    except Exception:
        return False
    return result.is_safe


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["ops"],
    summary="Liveness and readiness",
    responses={
        200: {"description": "Process is live; ready reflects self-check."},
        503: {"description": "Live but self-check failed.", "model": HealthResponse},
    },
)
def health() -> JSONResponse:
    """Liveness is the handler itself; readiness is a deterministic self-check."""
    ready = _readiness()
    payload = HealthResponse(
        status="ok",
        ready=ready,
        version=__version__,
        validators=["measurements", "laterality", "negations"],
    )
    return JSONResponse(status_code=200 if ready else 503, content=payload.model_dump())


@app.post(
    "/validate",
    response_model=ValidationResponse,
    tags=["validation"],
    summary="Validate an impression against findings",
    responses={
        200: {"description": "Deterministic validation result.", "model": ValidationResponse},
        413: {"description": "Request body too large.", "model": ErrorResponse},
        422: {"description": "Malformed request.", "model": ErrorResponse},
    },
)
def validate(request: ValidationRequest) -> ValidationResponse:
    """Run measurement, laterality, and negation checks on a report pair."""
    return evaluate(request.findings_text, request.impression_text)


def custom_openapi() -> dict[str, Any]:
    """Cache an OpenAPI schema with explicit error models."""
    if app.openapi_schema:
        return app.openapi_schema
    app.openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
        contact=app.contact,
        license_info=app.license_info,
    )
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore[method-assign]

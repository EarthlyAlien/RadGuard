"""Structured API error payloads."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    """Machine-readable error envelope returned by the HTTP layer."""

    code: str = Field(..., description="Stable error code, e.g. PAYLOAD_TOO_LARGE.")
    message: str = Field(..., description="Human-readable summary.")
    details: list[dict[str, Any]] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Top-level error document for OpenAPI and clients."""

    error: ErrorBody


def error_payload(code: str, message: str, details: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build a JSON-serializable error body."""
    return ErrorResponse(error=ErrorBody(code=code, message=message, details=details or [])).model_dump()

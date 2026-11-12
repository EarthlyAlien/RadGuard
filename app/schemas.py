"""Pydantic v2 request/response models for the RadGuard validation API."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.limits import MAX_TEXT_CHARS


class FlagCategory(str, Enum):
    """Hallucination categories detected by the deterministic validators."""

    MEASUREMENT_MISMATCH = "MEASUREMENT_MISMATCH"
    LATERALITY_CONFLICT = "LATERALITY_CONFLICT"
    NEGATION_FLIP = "NEGATION_FLIP"
    PARSE_AMBIGUITY = "PARSE_AMBIGUITY"


class FlagSeverity(str, Enum):
    """Clinical severity of a detected inconsistency."""

    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ValidationRequest(BaseModel):
    """LLM findings/impression pair submitted for guardrail checks."""

    model_config = ConfigDict(
        str_strip_whitespace=False,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "findings_text": (
                        "There is a 4 mm non-obstructing calculus in the left kidney. "
                        "No hydronephrosis."
                    ),
                    "impression_text": "4 cm stone in the right kidney.",
                }
            ]
        },
    )

    findings_text: str = Field(
        ...,
        min_length=0,
        max_length=MAX_TEXT_CHARS,
        description="Source radiologist findings / dictation.",
    )
    impression_text: str = Field(
        ...,
        min_length=0,
        max_length=MAX_TEXT_CHARS,
        description="Generated impression / summary to validate.",
    )

    @field_validator("findings_text", "impression_text")
    @classmethod
    def reject_null_bytes(cls, value: str) -> str:
        """Reject embedded NUL bytes that break downstream regex engines."""
        if "\x00" in value:
            raise ValueError("null bytes are not allowed")
        return value


class Flag(BaseModel):
    """A single deterministic inconsistency between findings and impression."""

    category: FlagCategory
    severity: FlagSeverity
    source_evidence: str
    generated_evidence: str
    explanation: str


class ValidationResponse(BaseModel):
    """Guardrail result. ``is_safe`` is true if and only if ``flags`` is empty."""

    is_safe: bool = Field(..., description="True iff flags is empty.")
    flags: list[Flag] = Field(
        ...,
        description="Ranked flags (CRITICAL before HIGH; measurement, laterality, negation, parse).",
    )
    latency_ms: float = Field(..., ge=0.0, description="Engine wall time in milliseconds.")


class HealthResponse(BaseModel):
    """Liveness and readiness document for orchestrators."""

    status: str = Field(..., description="Process liveness: always 'ok' if the handler ran.")
    ready: bool = Field(..., description="True when a self-check evaluation succeeds.")
    version: str
    validators: list[str]



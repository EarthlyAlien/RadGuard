"""Orchestration layer for RadGuard's deterministic validators."""

from __future__ import annotations

import logging
import time
from typing import Callable, Never

from app.limits import MAX_TEXT_CHARS
from app.normalize import normalize_text
from app.schemas import Flag, FlagCategory, FlagSeverity, ValidationResponse
from app.validators.laterality import validate_laterality
from app.validators.measurements import validate_measurements
from app.validators.negations import validate_negations

logger = logging.getLogger("radguard.engine")

ValidatorFn = Callable[[str, str], list[Flag]]

_VALIDATORS: tuple[tuple[str, ValidatorFn], ...] = (
    ("measurements", validate_measurements),
    ("laterality", validate_laterality),
    ("negations", validate_negations),
)


def _severity_rank(severity: FlagSeverity) -> int:
    """Rank severity so CRITICAL sorts before HIGH."""
    match severity:
        case FlagSeverity.CRITICAL:
            return 0
        case FlagSeverity.HIGH:
            return 1
        case _ as unreachable:
            never: Never = unreachable
            raise RuntimeError(f"unhandled severity: {never}")


def _category_rank(category: FlagCategory) -> int:
    """Stable category order after severity."""
    match category:
        case FlagCategory.MEASUREMENT_MISMATCH:
            return 0
        case FlagCategory.LATERALITY_CONFLICT:
            return 1
        case FlagCategory.NEGATION_FLIP:
            return 2
        case FlagCategory.PARSE_AMBIGUITY:
            return 3
        case _ as unreachable:
            never: Never = unreachable
            raise RuntimeError(f"unhandled category: {never}")


def rank_flags(flags: list[Flag]) -> list[Flag]:
    """Dedupe overlapping identical flags and rank CRITICAL first."""
    seen: set[tuple[str, str, str, str, str]] = set()
    unique: list[Flag] = []
    for flag in flags:
        key = (
            flag.category.value,
            flag.severity.value,
            flag.source_evidence,
            flag.generated_evidence,
            flag.explanation,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(flag)
    unique.sort(
        key=lambda flag: (
            _severity_rank(flag.severity),
            _category_rank(flag.category),
            flag.explanation,
        )
    )
    return unique


def _ambiguity_flag(source: str, generated: str, explanation: str) -> Flag:
    return Flag(
        category=FlagCategory.PARSE_AMBIGUITY,
        severity=FlagSeverity.HIGH,
        source_evidence=source,
        generated_evidence=generated,
        explanation=explanation,
    )


def _run_validator(name: str, fn: ValidatorFn, findings: str, impression: str) -> list[Flag]:
    """Run one validator; fail closed (flag, do not crash) on unexpected errors."""
    try:
        return list(fn(findings, impression))
    except Exception:
        logger.exception("Validator %s failed closed", name)
        return [
            _ambiguity_flag(
                name,
                name,
                f"Validator '{name}' encountered an internal error and failed closed.",
            )
        ]


def _clip(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_TEXT_CHARS:
        return text, False
    return text[:MAX_TEXT_CHARS], True


def collect_flags(findings_text: str, impression_text: str) -> list[Flag]:
    """Run every validator and return the ranked, deduped flag list."""
    findings, findings_clipped = _clip(normalize_text(findings_text))
    impression, impression_clipped = _clip(normalize_text(impression_text))
    flags: list[Flag] = []
    if findings_clipped or impression_clipped:
        flags.append(
            _ambiguity_flag(
                f"{len(findings_text)} chars",
                f"{len(impression_text)} chars",
                "Input exceeded the character limit and was truncated; failing closed.",
            )
        )
    for name, fn in _VALIDATORS:
        flags.extend(_run_validator(name, fn, findings, impression))
    return rank_flags(flags)


def evaluate(findings_text: str, impression_text: str) -> ValidationResponse:
    """Validate an impression against findings and record latency in milliseconds."""
    started = time.perf_counter()
    try:
        flags = collect_flags(findings_text, impression_text)
    except Exception:
        logger.exception("Engine failed closed")
        flags = [
            _ambiguity_flag(
                "engine",
                "engine",
                "Validation engine encountered an internal error and failed closed.",
            )
        ]
    latency_ms = (time.perf_counter() - started) * 1000.0
    return ValidationResponse(
        is_safe=len(flags) == 0,
        flags=flags,
        latency_ms=latency_ms,
    )

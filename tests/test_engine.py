"""Engine-level tests: ranking, empty/unicode/long input, fail-closed, latency."""

from __future__ import annotations

import math

from app.engine import evaluate, rank_flags
from app.limits import LATENCY_BUDGET_MS, MAX_TEXT_CHARS
from app.schemas import Flag, FlagCategory, FlagSeverity


def test_empty_and_whitespace_inputs_are_safe() -> None:
    assert evaluate("", "").is_safe is True
    assert evaluate("   \n\t  ", " ").is_safe is True


def test_mixed_case_and_unicode_nbsp_unit_conversion() -> None:
    result = evaluate(
        "THERE IS A 10\u00a0MM NODULE. NO PNEUMOTHORAX.",
        "1 cm nodule. No pneumothorax.",
    )
    assert result.is_safe is True


def test_very_long_input_does_not_crash() -> None:
    padding = ("unremarkable parenchyma. " * 2000)[: MAX_TEXT_CHARS + 500]
    result = evaluate(padding + " 10 mm nodule.", "1 cm nodule.")
    assert result.latency_ms >= 0.0
    assert any(flag.category == FlagCategory.PARSE_AMBIGUITY for flag in result.flags)
    assert result.is_safe is False


def test_garbage_input_does_not_crash() -> None:
    result = evaluate("??? ### \x08 \u2603", "@@@")
    assert isinstance(result.is_safe, bool)
    assert result.latency_ms >= 0.0


def test_rank_flags_critical_before_high_and_dedupes() -> None:
    high = Flag(
        category=FlagCategory.NEGATION_FLIP,
        severity=FlagSeverity.HIGH,
        source_evidence="a",
        generated_evidence="b",
        explanation="high",
    )
    critical = Flag(
        category=FlagCategory.MEASUREMENT_MISMATCH,
        severity=FlagSeverity.CRITICAL,
        source_evidence="c",
        generated_evidence="d",
        explanation="critical",
    )
    duplicate = Flag(
        category=FlagCategory.MEASUREMENT_MISMATCH,
        severity=FlagSeverity.CRITICAL,
        source_evidence="c",
        generated_evidence="d",
        explanation="critical",
    )
    ranked = rank_flags([high, critical, duplicate])
    assert [flag.severity for flag in ranked] == [FlagSeverity.CRITICAL, FlagSeverity.HIGH]
    assert len(ranked) == 2


def test_overlapping_measurement_and_laterality_both_surface() -> None:
    result = evaluate(
        "4 mm calculus in the left kidney.",
        "4 cm calculus in the right kidney.",
    )
    categories = [flag.category for flag in result.flags]
    assert FlagCategory.MEASUREMENT_MISMATCH in categories
    assert FlagCategory.LATERALITY_CONFLICT in categories
    assert categories.index(FlagCategory.MEASUREMENT_MISMATCH) < categories.index(
        FlagCategory.LATERALITY_CONFLICT
    )


def test_synthetic_cases_match_expected_safe(synthetic_cases: list[dict[str, object]]) -> None:
    mismatches: list[str] = []
    for case in synthetic_cases:
        result = evaluate(str(case["findings"]), str(case["impression"]))
        if result.is_safe is not bool(case["expected_safe"]):
            mismatches.append(
                f"{case['id']}: expected_safe={case['expected_safe']} got is_safe={result.is_safe} "
                f"flags={[flag.category.value for flag in result.flags]}"
            )
    assert mismatches == []


def test_latency_p99_under_budget(synthetic_cases: list[dict[str, object]]) -> None:
    samples: list[float] = []
    for case in synthetic_cases:
        for _ in range(25):
            result = evaluate(str(case["findings"]), str(case["impression"]))
            samples.append(result.latency_ms)
    ordered = sorted(samples)
    rank = (len(ordered) - 1) * 0.99
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        p99 = ordered[low]
    else:
        weight = rank - low
        p99 = ordered[low] * (1.0 - weight) + ordered[high] * weight
    assert p99 < LATENCY_BUDGET_MS, f"p99 {p99:.3f} ms exceeds {LATENCY_BUDGET_MS} ms"

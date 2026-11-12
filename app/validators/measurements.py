"""Measurement and unit consistency checks between findings and impression.

Extracts numeric values with units (mm, cm, %), including spelled-out aliases,
hyphenated and compact forms, decimals, and ranges. Lengths canonicalize to
millimeters: ``10 mm`` vs ``1 cm`` is equivalent (safe); ``3 mm`` vs ``3 cm``
is CRITICAL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas import Flag, FlagCategory, FlagSeverity

_UNIT_TOKEN = (
    r"millimet(?:er|re)s?|centimet(?:er|re)s?|"
    r"percent(?:age)?|pct|"
    r"mms|cms|mm|cm|%"
)

_RANGE_RE = re.compile(
    rf"(?P<low>\d*\.\d+|\d+)\s*(?:-|to|x)\s*(?P<high>\d*\.\d+|\d+)\s*-?\s*(?P<unit>{_UNIT_TOKEN})(?!\w)",
    re.IGNORECASE,
)
_SINGLE_RE = re.compile(
    rf"(?P<value>\d*\.\d+|\d+)\s*-?\s*(?P<unit>{_UNIT_TOKEN})(?!\w)",
    re.IGNORECASE,
)

_MM_UNITS = {"mm", "mms", "millimeter", "millimeters", "millimetre", "millimetres"}
_CM_UNITS = {"cm", "cms", "centimeter", "centimeters", "centimetre", "centimetres"}
_PCT_UNITS = {"%", "percent", "percentage", "pct"}

_CANONICAL_TOLERANCE_MM = 0.051
_CANONICAL_RELATIVE_TOLERANCE = 1e-3


@dataclass(frozen=True, slots=True)
class Measurement:
    """A single extracted measurement with a canonical (mm or percent) value."""

    value: float
    unit: str
    canonical_value: float
    canonical_unit: str
    raw: str
    start: int
    end: int


def canonicalize_unit(unit: str) -> str | None:
    """Return ``mm``, ``cm``, or ``%``, or None if the token is not a known unit."""
    token = unit.strip().lower().rstrip(".")
    if token in _MM_UNITS:
        return "mm"
    if token in _CM_UNITS:
        return "cm"
    if token in _PCT_UNITS:
        return "%"
    return None


def to_canonical(value: float, unit: str) -> tuple[float, str]:
    """Normalize a length to millimeters; leave percent unchanged.

    Returns:
        Tuple of ``(canonical_value, canonical_unit)`` where canonical_unit is
        ``mm`` for lengths and ``%`` for percentages.

    Raises:
        ValueError: If the unit is not a supported measurement unit.
    """
    canonical_unit = canonicalize_unit(unit)
    if canonical_unit is None:
        raise ValueError(f"Unsupported measurement unit: {unit!r}")
    if canonical_unit == "cm":
        return value * 10.0, "mm"
    if canonical_unit == "mm":
        return value, "mm"
    return value, "%"


def values_equivalent(left: float, right: float) -> bool:
    """Return True if two canonical values match within absolute/relative tolerance."""
    delta = abs(left - right)
    if delta <= _CANONICAL_TOLERANCE_MM:
        return True
    scale = max(abs(left), abs(right), 1.0)
    return delta <= _CANONICAL_RELATIVE_TOLERANCE * scale


def _parse_number(raw: str) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _build(value: float, unit_raw: str, raw: str, start: int, end: int) -> Measurement | None:
    unit = canonicalize_unit(unit_raw)
    if unit is None:
        return None
    canonical_value, canonical_unit = to_canonical(value, unit)
    return Measurement(
        value=value,
        unit=unit,
        canonical_value=canonical_value,
        canonical_unit=canonical_unit,
        raw=raw.strip(),
        start=start,
        end=end,
    )


def _overlaps(start: int, end: int, occupied: list[tuple[int, int]]) -> bool:
    for occupied_start, occupied_end in occupied:
        if start < occupied_end and end > occupied_start:
            return True
    return False


def extract_measurements(text: str) -> list[Measurement]:
    """Extract ordered measurements, expanding ranges into endpoint measurements."""
    if not text:
        return []
    measurements: list[Measurement] = []
    occupied: list[tuple[int, int]] = []

    for match in _RANGE_RE.finditer(text):
        low = _parse_number(match.group("low"))
        high = _parse_number(match.group("high"))
        if low is None or high is None:
            continue
        unit_raw = match.group("unit")
        raw = match.group(0).strip()
        span = (match.start(), match.end())
        low_item = _build(low, unit_raw, raw, span[0], span[1])
        high_item = _build(high, unit_raw, raw, span[0], span[1])
        if low_item is None or high_item is None:
            continue
        measurements.extend((low_item, high_item))
        occupied.append(span)

    for match in _SINGLE_RE.finditer(text):
        if _overlaps(match.start(), match.end(), occupied):
            continue
        value = _parse_number(match.group("value"))
        if value is None:
            continue
        item = _build(
            value,
            match.group("unit"),
            match.group(0),
            match.start(),
            match.end(),
        )
        if item is None:
            continue
        measurements.append(item)

    measurements.sort(key=lambda item: (item.start, item.end, item.value))
    return measurements


def _same_dimension(left: Measurement, right: Measurement) -> bool:
    return left.canonical_unit == right.canonical_unit


def _magnitude_error(source: Measurement, generated: Measurement) -> bool:
    """True when the same displayed number was restated with an incompatible unit."""
    if not _same_dimension(source, generated):
        return False
    if source.unit == generated.unit:
        return False
    return abs(source.value - generated.value) <= _CANONICAL_TOLERANCE_MM and not values_equivalent(
        source.canonical_value, generated.canonical_value
    )


def _flag(
    severity: FlagSeverity,
    source_evidence: str,
    generated_evidence: str,
    explanation: str,
) -> Flag:
    return Flag(
        category=FlagCategory.MEASUREMENT_MISMATCH,
        severity=severity,
        source_evidence=source_evidence,
        generated_evidence=generated_evidence,
        explanation=explanation,
    )


def validate_measurements(findings_text: str, impression_text: str) -> list[Flag]:
    """Compare impression measurements against findings after unit normalization."""
    source = extract_measurements(findings_text)
    generated = extract_measurements(impression_text)
    flags: list[Flag] = []
    used_source: set[int] = set()
    seen_pairs: set[tuple[str, str]] = set()

    def claim_canonical_match(item: Measurement) -> int | None:
        for index, candidate in enumerate(source):
            if index in used_source or not _same_dimension(item, candidate):
                continue
            if values_equivalent(item.canonical_value, candidate.canonical_value):
                return index
        return None

    unmatched: list[Measurement] = []
    for item in generated:
        match_index = claim_canonical_match(item)
        if match_index is None:
            unmatched.append(item)
        else:
            used_source.add(match_index)

    remaining_source = [item for index, item in enumerate(source) if index not in used_source]

    def emit(flag: Flag) -> None:
        key = (flag.source_evidence, flag.generated_evidence)
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        flags.append(flag)

    for item in unmatched:
        unit_conflict = next((candidate for candidate in remaining_source if _magnitude_error(candidate, item)), None)
        if unit_conflict is not None:
            remaining_source.remove(unit_conflict)
            emit(
                _flag(
                    FlagSeverity.CRITICAL,
                    unit_conflict.raw,
                    item.raw,
                    (
                        f"Measurement unit/value mismatch: {item.canonical_value:g}"
                        f"{item.canonical_unit} generated vs {unit_conflict.canonical_value:g}"
                        f"{unit_conflict.canonical_unit} in source."
                    ),
                )
            )
            continue

        same_type = next((candidate for candidate in remaining_source if _same_dimension(candidate, item)), None)
        if same_type is not None:
            remaining_source.remove(same_type)
            emit(
                _flag(
                    FlagSeverity.CRITICAL,
                    same_type.raw,
                    item.raw,
                    (
                        f"Measurement value mismatch: {item.canonical_value:g}"
                        f"{item.canonical_unit} generated vs {same_type.canonical_value:g}"
                        f"{same_type.canonical_unit} in source."
                    ),
                )
            )
            continue

        emit(
            _flag(
                FlagSeverity.HIGH,
                "(no matching source measurement)",
                item.raw,
                f"Impression introduces measurement {item.raw} with no equivalent in the findings.",
            )
        )

    return flags

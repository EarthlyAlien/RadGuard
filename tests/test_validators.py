"""Unit, golden, and property-style tests for the deterministic validators."""

from __future__ import annotations

import pytest

from app.engine import evaluate
from app.schemas import FlagCategory, FlagSeverity
from app.validators.laterality import extract_laterality_mentions, validate_laterality
from app.validators.measurements import (
    extract_measurements,
    to_canonical,
    validate_measurements,
    values_equivalent,
)
from app.validators.negations import (
    Polarity,
    extract_entity_mentions,
    is_negated,
    mention_polarity,
    validate_negations,
)


def test_cm_to_mm_normalization_is_equivalent() -> None:
    """10 mm and 1 cm must be treated as the same length."""
    ten_mm, unit_mm = to_canonical(10.0, "mm")
    one_cm, unit_cm = to_canonical(1.0, "cm")
    assert unit_mm == unit_cm == "mm"
    assert values_equivalent(ten_mm, one_cm)
    assert values_equivalent(to_canonical(1.5, "cm")[0], 15.0)


def test_ten_mm_versus_one_cm_is_safe() -> None:
    flags = validate_measurements(
        "There is a 10 mm nodule in the right upper lobe.",
        "1 cm right upper lobe nodule.",
    )
    assert flags == []


def test_three_mm_versus_three_cm_is_critical() -> None:
    flags = validate_measurements(
        "A 3 mm ground-glass nodule is present.",
        "3 cm ground-glass nodule.",
    )
    assert len(flags) == 1
    assert flags[0].category == FlagCategory.MEASUREMENT_MISMATCH
    assert flags[0].severity == FlagSeverity.CRITICAL
    assert "3 mm" in flags[0].source_evidence
    assert "3 cm" in flags[0].generated_evidence


@pytest.mark.parametrize(
    ("source", "generated"),
    [
        ("10 mm", "1 cm"),
        ("10mm", "1.0 cm"),
        ("10-mm", "1 cm"),
        ("10 millimeters", "1 centimetre"),
        ("10 millimetres", "1 centimeter"),
        ("10.0 mm", "1.00 cm"),
        ("1.5 cm", "15 mm"),
        ("12 mm", "1.2 cm"),
        ("8-10 mm", "0.8-1.0 cm"),
        ("1.2 x 0.8 cm", "12 x 8 mm"),
        ("50%", "50 percent"),
        ("50 percent", "50%"),
    ],
)
def test_equivalent_measurements_are_safe(source: str, generated: str) -> None:
    flags = validate_measurements(
        f"Finding measures {source}.",
        f"Finding measures {generated}.",
    )
    assert flags == [], (source, generated, flags)


@pytest.mark.parametrize(
    ("source", "generated"),
    [
        ("3 mm", "3 cm"),
        ("3mm", "3cm"),
        ("3 millimeters", "3 centimeters"),
        ("3.0 mm", "3.0 cm"),
        ("4 mm", "4 cm"),
        ("4-5 mm", "4-5 cm"),
        ("50%", "70%"),
    ],
)
def test_incompatible_measurements_are_flagged(source: str, generated: str) -> None:
    flags = validate_measurements(
        f"Finding measures {source}.",
        f"Finding measures {generated}.",
    )
    assert flags
    assert flags[0].category == FlagCategory.MEASUREMENT_MISMATCH
    assert flags[0].severity == FlagSeverity.CRITICAL


def test_percent_stenosis_match_is_safe() -> None:
    source = extract_measurements("There is 50% stenosis of the left proximal ICA.")
    assert len(source) == 1
    assert source[0].canonical_unit == "%"
    assert source[0].canonical_value == 50.0
    flags = validate_measurements(
        "There is 50% stenosis of the left proximal ICA.",
        "50% stenosis of the left ICA.",
    )
    assert flags == []


def test_extract_measurements_handles_compact_units() -> None:
    extracted = extract_measurements("Lesion measures 12mm by 8 mm (0.8 cm).")
    assert [item.canonical_unit for item in extracted] == ["mm", "mm", "mm"]
    assert values_equivalent(extracted[0].canonical_value, 12.0)
    assert values_equivalent(extracted[2].canonical_value, 8.0)


def test_extract_range_expands_endpoints() -> None:
    extracted = extract_measurements("Measures 3-4 mm.")
    assert len(extracted) == 2
    assert {item.canonical_value for item in extracted} == {3.0, 4.0}


def test_laterality_swap_is_critical() -> None:
    flags = validate_laterality(
        "There is a 4 mm calculus in the left kidney.",
        "4 mm stone in the right kidney.",
    )
    assert len(flags) == 1
    assert flags[0].category == FlagCategory.LATERALITY_CONFLICT
    assert flags[0].severity == FlagSeverity.CRITICAL


def test_laterality_introduction_is_flagged() -> None:
    flags = validate_laterality(
        "There is a nodule in the lung. No effusion.",
        "Right lung nodule. No effusion.",
    )
    assert flags
    assert flags[0].category == FlagCategory.LATERALITY_CONFLICT
    assert "introduced without source support" in flags[0].explanation


def test_bilateral_covers_left_and_right() -> None:
    flags = validate_laterality(
        "Small bilateral pleural effusions.",
        "Small left pleural effusion.",
    )
    assert flags == []


def test_unilateral_to_bilateral_is_flagged() -> None:
    flags = validate_laterality(
        "Small left pleural effusion.",
        "Bilateral pleural effusions.",
    )
    assert flags
    assert flags[0].category == FlagCategory.LATERALITY_CONFLICT


def test_renal_synonym_matches_kidney() -> None:
    mentions = extract_laterality_mentions("4 mm left renal calculus.")
    assert any(item.organ == "kidney" and item.side == "left" for item in mentions)
    flags = validate_laterality(
        "4 mm calculus in the left kidney.",
        "4 mm left renal calculus.",
    )
    assert flags == []


def test_lobe_abbreviation_matches_right_lower_lobe() -> None:
    flags = validate_laterality(
        "Consolidation in the right lower lobe.",
        "RLL pneumonia.",
    )
    assert flags == []


def test_on_the_left_binds_kidney() -> None:
    flags = validate_laterality(
        "There is a 4 mm calculus on the left kidney.",
        "4 mm calculus on the right kidney.",
    )
    assert flags
    assert flags[0].category == FlagCategory.LATERALITY_CONFLICT


def test_parenthetical_laterality() -> None:
    flags = validate_laterality(
        "Calculus in the kidney (L).",
        "Left renal calculus.",
    )
    assert flags == []


def test_unbound_laterality_swap() -> None:
    flags = validate_laterality(
        "Left-sided pain. No fracture.",
        "Right-sided pain. No fracture.",
    )
    assert flags
    assert flags[0].category == FlagCategory.LATERALITY_CONFLICT


def test_lingula_implies_left_lung() -> None:
    flags = validate_laterality(
        "Opacity in the lingula.",
        "Left lung opacity.",
    )
    assert flags == []


@pytest.mark.parametrize("organ", ["kidney", "lung", "breast", "wrist", "thyroid"])
def test_left_right_swap_is_flagged_for_organs(organ: str) -> None:
    flags = validate_laterality(f"Finding in the left {organ}.", f"Finding in the right {organ}.")
    assert flags
    assert all(flag.category == FlagCategory.LATERALITY_CONFLICT for flag in flags)


def test_negation_drop_fracture_is_critical() -> None:
    flags = validate_negations(
        "There is no acute fracture of the distal radius.",
        "Acute fracture of the distal radius.",
    )
    assert len(flags) == 1
    assert flags[0].category == FlagCategory.NEGATION_FLIP
    assert flags[0].severity == FlagSeverity.CRITICAL


def test_negation_drop_pneumothorax_is_flagged() -> None:
    flags = validate_negations(
        "The lungs are clear. There is no pneumothorax.",
        "Small pneumothorax.",
    )
    assert flags
    assert flags[0].category == FlagCategory.NEGATION_FLIP


def test_both_negated_is_safe() -> None:
    flags = validate_negations(
        "There is no pneumothorax or pleural effusion.",
        "No pneumothorax. No effusion.",
    )
    assert flags == []


def test_positive_to_negated_is_flagged() -> None:
    flags = validate_negations(
        "There is a moderate pleural effusion.",
        "No pleural effusion.",
    )
    assert len(flags) == 1
    assert "present in the findings but negated" in flags[0].explanation


def test_rule_out_is_treated_as_negated() -> None:
    text = "Rule out pneumothorax before discharge."
    mentions = extract_entity_mentions(text)
    assert mentions
    assert all(item.polarity is Polarity.NEGATED for item in mentions if item.canonical == "pneumothorax")
    flags = validate_negations(text, "Small pneumothorax.")
    assert flags
    assert flags[0].category == FlagCategory.NEGATION_FLIP


def test_negative_for_window() -> None:
    text = "The study is negative for pneumothorax."
    start = text.lower().index("pneumothorax")
    assert is_negated(text, start, start + len("pneumothorax"))


def test_without_scopes_effusion() -> None:
    flags = validate_negations(
        "Pneumonia without pleural effusion.",
        "Pneumonia with pleural effusion.",
    )
    assert flags
    assert flags[0].category == FlagCategory.NEGATION_FLIP


def test_mass_does_not_match_massive() -> None:
    mentions = extract_entity_mentions("Massive cardiomegaly without a pulmonary mass.")
    assert [item.canonical for item in mentions] == ["mass"]
    assert mentions[0].polarity is Polarity.NEGATED


def test_denies_and_unremarkable_for_and_free_of() -> None:
    assert mention_polarity(*_span("The patient denies fracture.")) is Polarity.NEGATED
    assert mention_polarity(*_span("Unremarkable for pneumothorax.")) is Polarity.NEGATED
    assert mention_polarity(*_span("Free of effusion.")) is Polarity.NEGATED


def test_cannot_exclude_is_uncertain() -> None:
    text = "Cannot exclude a small pneumothorax."
    start = text.lower().index("pneumothorax")
    assert mention_polarity(text, start, start + len("pneumothorax")) is Polarity.UNCERTAIN
    flags = validate_negations(text, "Small pneumothorax.")
    assert flags
    assert flags[0].severity == FlagSeverity.HIGH


def test_double_negative_not_without_is_asserted() -> None:
    text = "The study is not without pleural effusion."
    start = text.lower().index("effusion")
    assert mention_polarity(text, start, start + len("effusion")) is Polarity.ASSERTED
    flags = validate_negations(text, "Pleural effusion is present.")
    assert flags == []


def test_sentence_boundary_blocks_negation_leak() -> None:
    text = "There is no pneumothorax. Small effusion is present."
    effusion_at = text.lower().index("effusion")
    assert mention_polarity(text, effusion_at, effusion_at + len("effusion")) is Polarity.ASSERTED


@pytest.mark.parametrize(
    "entity",
    ["fracture", "pneumothorax", "effusion", "mass", "hemorrhage", "stenosis"],
)
def test_negation_drop_is_flagged_for_every_entity(entity: str) -> None:
    flags = validate_negations(f"No {entity} identified.", f"Acute {entity} is present.")
    assert flags
    assert flags[0].category == FlagCategory.NEGATION_FLIP
    assert flags[0].severity == FlagSeverity.CRITICAL


def test_golden_measurement_mismatch() -> None:
    result = evaluate(
        "A 3 mm ground-glass nodule is present.",
        "3 cm ground-glass nodule.",
    )
    assert result.is_safe is False
    assert result.flags[0].category == FlagCategory.MEASUREMENT_MISMATCH
    assert result.flags[0].severity == FlagSeverity.CRITICAL


def test_golden_laterality_conflict() -> None:
    result = evaluate(
        "4 mm calculus in the left kidney.",
        "4 mm calculus in the right kidney.",
    )
    assert result.is_safe is False
    assert any(flag.category == FlagCategory.LATERALITY_CONFLICT for flag in result.flags)


def test_golden_negation_flip() -> None:
    result = evaluate(
        "There is no acute fracture.",
        "Acute fracture.",
    )
    assert result.is_safe is False
    assert any(flag.category == FlagCategory.NEGATION_FLIP for flag in result.flags)


def test_engine_safe_unit_conversion_end_to_end() -> None:
    result = evaluate(
        "There is a 10 mm nodule in the right upper lobe. No pneumothorax.",
        "1 cm right upper lobe nodule. No pneumothorax.",
    )
    assert result.is_safe is True
    assert result.flags == []
    assert result.latency_ms >= 0.0


def test_engine_readme_example_flags_measurement_and_laterality() -> None:
    result = evaluate(
        "There is a 4 mm non-obstructing calculus in the left kidney. No hydronephrosis.",
        "4 cm stone in the right kidney.",
    )
    assert result.is_safe is False
    categories = {flag.category for flag in result.flags}
    assert FlagCategory.MEASUREMENT_MISMATCH in categories
    assert FlagCategory.LATERALITY_CONFLICT in categories
    assert _severity_order(result.flags)


def _severity_order(flags: list) -> bool:
    ranks = {"CRITICAL": 0, "HIGH": 1}
    values = [ranks[flag.severity.value] for flag in flags]
    return values == sorted(values)


def _span(text: str) -> tuple[str, int, int]:
    mentions = extract_entity_mentions(text)
    assert mentions
    mention = mentions[0]
    return text, mention.start, mention.end

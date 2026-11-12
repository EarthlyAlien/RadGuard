"""Laterality consistency checks between findings and impression.

Flags an impression that swaps left/right for the same organ, or that
introduces a laterality the findings never assigned to that structure.
``bilateral`` in the source covers both left and right. Unbound phrases
such as ``left-sided`` / ``on the right`` are associated with nearby organs
or treated as a generic laterality token.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas import Flag, FlagCategory, FlagSeverity

UNBOUND_ORGAN = "*"

ORGAN_GROUPS: dict[str, str] = {
    "kidney": "kidney",
    "kidneys": "kidney",
    "renal": "kidney",
    "ureter": "ureter",
    "ureters": "ureter",
    "lung": "lung",
    "lungs": "lung",
    "pulmonary": "lung",
    "hemithorax": "lung",
    "upper lobe": "lung",
    "middle lobe": "lung",
    "lower lobe": "lung",
    "lingula": "lung",
    "lingular": "lung",
    "pleura": "pleura",
    "pleural": "pleura",
    "breast": "breast",
    "breasts": "breast",
    "liver": "liver",
    "hepatic": "liver",
    "thyroid": "thyroid",
    "adrenal": "adrenal",
    "adrenals": "adrenal",
    "ovary": "ovary",
    "ovaries": "ovary",
    "ovarian": "ovary",
    "testis": "testis",
    "testes": "testis",
    "testicle": "testis",
    "testicular": "testis",
    "hemisphere": "hemisphere",
    "cerebral hemisphere": "hemisphere",
    "ica": "carotid",
    "carotid": "carotid",
    "mca": "mca",
    "aca": "aca",
    "pca": "pca",
    "ventricle": "ventricle",
    "ventricles": "ventricle",
    "atrium": "atrium",
    "atria": "atrium",
    "hip": "hip",
    "hips": "hip",
    "knee": "knee",
    "knees": "knee",
    "shoulder": "shoulder",
    "shoulders": "shoulder",
    "wrist": "wrist",
    "wrists": "wrist",
    "ankle": "ankle",
    "ankles": "ankle",
    "femur": "femur",
    "femoral": "femur",
    "tibia": "tibia",
    "tibial": "tibia",
    "fibula": "fibula",
    "radius": "radius",
    "radial": "radius",
    "ulna": "ulna",
    "ulnar": "ulna",
    "humerus": "humerus",
    "clavicle": "clavicle",
    "scapula": "scapula",
    "orbit": "orbit",
    "orbits": "orbit",
    "optic": "optic",
    "frontal": "frontal",
    "parietal": "parietal",
    "temporal": "temporal",
    "occipital": "occipital",
    "cerebellum": "cerebellum",
    "cerebellar": "cerebellum",
    "maxillary": "maxillary",
    "sinus": "sinus",
    "sinuses": "sinus",
    "axilla": "axilla",
    "axillary": "axilla",
    "iliac": "iliac",
    "parotid": "parotid",
    "tonsil": "tonsil",
    "tonsils": "tonsil",
    "hippocampus": "hippocampus",
    "hippocampal": "hippocampus",
}

_LOBE_ABBREVIATIONS: dict[str, tuple[str, str]] = {
    "rul": ("right", "lung"),
    "rml": ("right", "lung"),
    "rll": ("right", "lung"),
    "lul": ("left", "lung"),
    "lll": ("left", "lung"),
}

_IMPLIED_SIDE: dict[str, tuple[str, str]] = {
    "lingula": ("left", "lung"),
    "lingular": ("left", "lung"),
}

_SIDE_RE = re.compile(
    r"\b(?:on\s+the\s+(?P<on_side>left|right)"
    r"|(?P<sided>left|right)\s*-?\s*sided"
    r"|(?P<bilat>bilateral(?:ly)?)"
    r"|(?P<side>left|right))\b",
    re.IGNORECASE,
)
_ABBREV_RE = re.compile(r"\b(RUL|RML|RLL|LUL|LLL)\b")
_PAREN_SIDE_RE = re.compile(
    r"\b(?P<phrase>"
    + "|".join(re.escape(term) for term in sorted(ORGAN_GROUPS, key=len, reverse=True))
    + r")\s*\(\s*(?P<paren>L|R|Lt|Rt|left|right)\s*\)",
    re.IGNORECASE,
)
_ORGAN_TERMS = sorted(ORGAN_GROUPS.keys(), key=len, reverse=True)
_ORGAN_RE = re.compile(
    r"\b(" + "|".join(re.escape(term) for term in _ORGAN_TERMS) + r")\b",
    re.IGNORECASE,
)

_WINDOW_CHARS = 64


@dataclass(frozen=True, slots=True)
class LateralityMention:
    """An anatomical structure tagged with a laterality."""

    organ: str
    side: str
    evidence: str


def normalize_side(token: str) -> str | None:
    """Map a laterality token to ``left``, ``right``, or ``bilateral``."""
    lowered = token.lower().strip().rstrip(".")
    if lowered in {"l", "lt"} or lowered.startswith("left"):
        return "left"
    if lowered in {"r", "rt"} or lowered.startswith("right"):
        return "right"
    if lowered.startswith("bilateral"):
        return "bilateral"
    return None


def _window(text: str, start: int, end: int) -> str:
    return text[max(0, start - _WINDOW_CHARS) : min(len(text), end + _WINDOW_CHARS)]


def _organs_in(text: str) -> list[tuple[str, str]]:
    """Return ``(canonical_organ, matched_phrase)`` pairs in ``text``."""
    found: list[tuple[str, str]] = []
    for match in _ORGAN_RE.finditer(text):
        phrase = match.group(1).lower()
        group = ORGAN_GROUPS.get(phrase)
        if group is None:
            continue
        found.append((group, match.group(1)))
    return found


def _side_from_match(match: re.Match[str]) -> str | None:
    for key in ("on_side", "sided", "bilat", "side"):
        token = match.group(key)
        if token:
            return normalize_side(token)
    return None


def extract_laterality_mentions(text: str) -> list[LateralityMention]:
    """Extract (organ, side) mentions, including lobe abbreviations and unbound sides."""
    if not text:
        return []
    mentions: list[LateralityMention] = []
    seen: set[tuple[str, str, str]] = set()

    def add(organ: str, side: str, evidence: str) -> None:
        key = (organ, side, evidence.lower())
        if key in seen:
            return
        seen.add(key)
        mentions.append(LateralityMention(organ=organ, side=side, evidence=evidence))

    for match in _SIDE_RE.finditer(text):
        side = _side_from_match(match)
        if side is None:
            continue
        snippet = _window(text, match.start(), match.end())
        organs = _organs_in(snippet)
        if not organs:
            add(UNBOUND_ORGAN, side, match.group(0))
            continue
        for organ, phrase in organs:
            evidence = f"{side} {phrase}".strip()
            add(organ, side, evidence)

    for match in _ABBREV_RE.finditer(text):
        side, organ = _LOBE_ABBREVIATIONS[match.group(1).lower()]
        add(organ, side, match.group(1))

    for match in _PAREN_SIDE_RE.finditer(text):
        phrase = match.group("phrase").lower()
        organ = ORGAN_GROUPS.get(phrase)
        side = normalize_side(match.group("paren"))
        if organ is None or side is None:
            continue
        add(organ, side, match.group(0))

    for match in _ORGAN_RE.finditer(text):
        phrase = match.group(1).lower()
        implied = _IMPLIED_SIDE.get(phrase)
        if implied is None:
            continue
        side, organ = implied
        add(organ, side, match.group(1))

    return mentions


def organs_mentioned(text: str) -> set[str]:
    """Return canonical organ groups mentioned anywhere in ``text``."""
    if not text:
        return set()
    found: set[str] = set()
    for match in _ORGAN_RE.finditer(text):
        group = ORGAN_GROUPS.get(match.group(1).lower())
        if group is not None:
            found.add(group)
    return found


def _sides_by_organ(mentions: list[LateralityMention]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for mention in mentions:
        grouped.setdefault(mention.organ, set()).add(mention.side)
    return grouped


def _all_sides(mentions: list[LateralityMention]) -> set[str]:
    sides: set[str] = set()
    for mention in mentions:
        if mention.side == "bilateral":
            sides.update({"left", "right", "bilateral"})
        else:
            sides.add(mention.side)
    return sides


def _evidence_for(mentions: list[LateralityMention], organ: str, side: str | None = None) -> str:
    for mention in mentions:
        if mention.organ != organ:
            continue
        if side is None or mention.side == side:
            return mention.evidence
    return organ


def _sides_compatible(source_sides: set[str], generated_side: str) -> bool:
    """Return True if the generated laterality is supported by the source set."""
    if generated_side in source_sides:
        return True
    if "bilateral" in source_sides and generated_side in {"left", "right"}:
        return True
    if generated_side == "bilateral" and source_sides.issuperset({"left", "right"}):
        return True
    return False


def validate_laterality(findings_text: str, impression_text: str) -> list[Flag]:
    """Flag laterality swaps and laterality introduced without source support."""
    source_mentions = extract_laterality_mentions(findings_text)
    generated_mentions = extract_laterality_mentions(impression_text)
    source_sides = _sides_by_organ(source_mentions)
    generated_sides = _sides_by_organ(generated_mentions)
    source_organs = organs_mentioned(findings_text)
    source_global = _all_sides(source_mentions)

    flags: list[Flag] = []
    flagged: set[tuple[str, str]] = set()

    for organ, sides in generated_sides.items():
        for side in sorted(sides):
            key = (organ, side)
            if key in flagged:
                continue
            src = source_sides.get(organ, set())
            if _sides_compatible(src, side):
                continue
            if organ == UNBOUND_ORGAN:
                opposite = {"left": "right", "right": "left"}.get(side)
                unbound_src = source_sides.get(UNBOUND_ORGAN, set())
                if opposite and opposite in unbound_src and side not in unbound_src and "bilateral" not in unbound_src:
                    flagged.add(key)
                    flags.append(
                        Flag(
                            category=FlagCategory.LATERALITY_CONFLICT,
                            severity=FlagSeverity.CRITICAL,
                            source_evidence=_evidence_for(source_mentions, UNBOUND_ORGAN, opposite),
                            generated_evidence=_evidence_for(generated_mentions, UNBOUND_ORGAN, side),
                            explanation=(
                                f"Laterality conflict detected: '{side}' introduced without source support."
                            ),
                        )
                    )
                continue
            if not src and organ not in source_organs:
                continue

            flagged.add(key)
            if src & {"left", "right", "bilateral"} and not _sides_compatible(src, side):
                source_side = sorted(src)[0]
                flags.append(
                    Flag(
                        category=FlagCategory.LATERALITY_CONFLICT,
                        severity=FlagSeverity.CRITICAL,
                        source_evidence=_evidence_for(source_mentions, organ, source_side),
                        generated_evidence=_evidence_for(generated_mentions, organ, side),
                        explanation=(
                            f"Laterality conflict detected: '{side}' {organ} in the impression "
                            f"conflicts with '{source_side}' {organ} in the findings."
                        ),
                    )
                )
            else:
                flags.append(
                    Flag(
                        category=FlagCategory.LATERALITY_CONFLICT,
                        severity=FlagSeverity.HIGH,
                        source_evidence=_evidence_for(source_mentions, organ) if src else organ,
                        generated_evidence=_evidence_for(generated_mentions, organ, side),
                        explanation=(
                            f"Laterality conflict detected: '{side}' introduced without source support."
                        ),
                    )
                )

    for organ, sides in generated_sides.items():
        if organ == UNBOUND_ORGAN:
            continue
        for side in sides:
            if side not in {"left", "right"}:
                continue
            if (organ, side) in flagged:
                continue
            if side in source_global or "bilateral" in source_global:
                continue
            if organ not in source_organs:
                continue
            flags.append(
                Flag(
                    category=FlagCategory.LATERALITY_CONFLICT,
                    severity=FlagSeverity.HIGH,
                    source_evidence=organ,
                    generated_evidence=_evidence_for(generated_mentions, organ, side),
                    explanation=(
                        f"Laterality conflict detected: '{side}' introduced without source support."
                    ),
                )
            )

    return flags

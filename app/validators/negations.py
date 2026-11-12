"""Window-based negation detector for high-risk radiology entities.

Compares polarity of ``fracture``, ``pneumothorax``, ``effusion``, ``mass``,
``hemorrhage``, and ``stenosis`` between findings and impression.

Polarity is three-state: asserted, negated, or uncertain (``cannot exclude``).
Negation cues include ``no``, ``without``, ``absent``, ``negative for``,
``rule out``, ``denies``, ``unremarkable for``, and ``free of``. Double
negatives such as ``not without`` cancel back to asserted. Sentence
boundaries stop cue scope from leaking into the next sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from app.schemas import Flag, FlagCategory, FlagSeverity

ENTITY_CANONICAL: dict[str, str] = {
    "fracture": "fracture",
    "fractures": "fracture",
    "pneumothorax": "pneumothorax",
    "pneumothoraces": "pneumothorax",
    "effusion": "effusion",
    "effusions": "effusion",
    "mass": "mass",
    "masses": "mass",
    "hemorrhage": "hemorrhage",
    "hemorrhages": "hemorrhage",
    "haemorrhage": "hemorrhage",
    "haemorrhages": "hemorrhage",
    "stenosis": "stenosis",
    "stenoses": "stenosis",
}


class Polarity(str, Enum):
    """Local polarity of a guarded entity mention."""

    ASSERTED = "asserted"
    NEGATED = "negated"
    UNCERTAIN = "uncertain"


_ENTITY_RE = re.compile(
    r"\b(" + "|".join(re.escape(term) for term in sorted(ENTITY_CANONICAL, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_DOUBLE_NEGATIVE_RE = re.compile(
    r"\bnot\s+(?:without|absent|free\s+of)\b",
    re.IGNORECASE,
)
_UNCERTAIN_RE = re.compile(
    r"\b("
    r"cannot\s+exclude|can\s+not\s+exclude|can't\s+exclude|"
    r"cannot\s+be\s+excluded|could\s+not\s+exclude|"
    r"cannot\s+rule\s+out|could\s+not\s+rule\s+out|"
    r"not\s+excluded|not\s+be\s+excluded"
    r")\b",
    re.IGNORECASE,
)
_PRE_NEGATION_RE = re.compile(
    r"\b("
    r"no|without|absent|denies|denied|"
    r"negative\s+for|"
    r"ruled\s+out|rule\s+out|"
    r"no\s+evidence\s+(?:of|for)|without\s+evidence\s+(?:of|for)|"
    r"free\s+of|absence\s+of|lack\s+of|clear\s+of|"
    r"unremarkable\s+for|no\s+definite|no\s+discrete|"
    r"failed\s+to\s+(?:demonstrate|identify|show)"
    r")\b",
    re.IGNORECASE,
)
_POST_NEGATION_RE = re.compile(
    r"^\s*(?:is|was|are|were)?\s*"
    r"(?:absent|not\s+(?:identified|seen|present|demonstrated|evident)|ruled\s+out)\b",
    re.IGNORECASE,
)
_POST_UNCERTAIN_RE = re.compile(
    r"^\s*(?:cannot\s+be\s+excluded|is\s+not\s+excluded|cannot\s+be\s+ruled\s+out)\b",
    re.IGNORECASE,
)

_PRE_WINDOW_CHARS = 80
_POST_WINDOW_CHARS = 48
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class EntityMention:
    """A single entity span with three-state polarity."""

    canonical: str
    raw: str
    polarity: Polarity
    start: int
    end: int


def _pre_scope(text: str, start: int) -> str:
    """Return the in-sentence window before ``start``."""
    pre = text[max(0, start - _PRE_WINDOW_CHARS) : start]
    parts = _SENTENCE_SPLIT_RE.split(pre)
    return parts[-1] if parts else pre


def _post_scope(text: str, end: int) -> str:
    """Return the in-sentence window after ``end``."""
    post = text[end : min(len(text), end + _POST_WINDOW_CHARS)]
    parts = _SENTENCE_SPLIT_RE.split(post)
    return parts[0] if parts else post


def mention_polarity(text: str, start: int, end: int) -> Polarity:
    """Classify polarity of the entity span using windowed cue detectors."""
    pre = _pre_scope(text, start)
    post = _post_scope(text, end)
    if _DOUBLE_NEGATIVE_RE.search(pre):
        return Polarity.ASSERTED
    if _UNCERTAIN_RE.search(pre) or _POST_UNCERTAIN_RE.search(post):
        return Polarity.UNCERTAIN
    if _PRE_NEGATION_RE.search(pre) or _POST_NEGATION_RE.search(post):
        return Polarity.NEGATED
    return Polarity.ASSERTED


def is_negated(text: str, start: int, end: int) -> bool:
    """Return True if a negation cue scopes this entity span (not uncertain)."""
    return mention_polarity(text, start, end) is Polarity.NEGATED


def extract_entity_mentions(text: str) -> list[EntityMention]:
    """Extract guarded entities and their local polarity."""
    if not text:
        return []
    mentions: list[EntityMention] = []
    for match in _ENTITY_RE.finditer(text):
        raw = match.group(1)
        canonical = ENTITY_CANONICAL.get(raw.lower())
        if canonical is None:
            continue
        mentions.append(
            EntityMention(
                canonical=canonical,
                raw=match.group(0),
                polarity=mention_polarity(text, match.start(), match.end()),
                start=match.start(),
                end=match.end(),
            )
        )
    return mentions


def entity_polarity(mentions: list[EntityMention]) -> dict[str, Polarity]:
    """Collapse mentions to one polarity per entity.

    Precedence: any ASSERTED mention wins, else UNCERTAIN, else NEGATED.
    Mixed "no X except X in Y" is therefore treated as present.
    """
    polarity: dict[str, Polarity] = {}
    for mention in mentions:
        current = polarity.get(mention.canonical)
        if current is None:
            polarity[mention.canonical] = mention.polarity
            continue
        if current is Polarity.ASSERTED:
            continue
        if mention.polarity is Polarity.ASSERTED:
            polarity[mention.canonical] = Polarity.ASSERTED
            continue
        if mention.polarity is Polarity.UNCERTAIN and current is Polarity.NEGATED:
            polarity[mention.canonical] = Polarity.UNCERTAIN
    return polarity


def _evidence(mentions: list[EntityMention], canonical: str, polarity: Polarity) -> str:
    for mention in mentions:
        if mention.canonical == canonical and mention.polarity is polarity:
            if polarity is Polarity.NEGATED:
                return f"no {mention.raw}"
            if polarity is Polarity.UNCERTAIN:
                return f"cannot exclude {mention.raw}"
            return mention.raw
    return canonical


def _flip_severity(source: Polarity, generated: Polarity) -> FlagSeverity:
    if source is Polarity.NEGATED and generated is Polarity.ASSERTED:
        return FlagSeverity.CRITICAL
    if source is Polarity.ASSERTED and generated is Polarity.NEGATED:
        return FlagSeverity.CRITICAL
    return FlagSeverity.HIGH


def _flip_explanation(entity: str, source: Polarity, generated: Polarity) -> str:
    if source is Polarity.NEGATED and generated is Polarity.ASSERTED:
        return (
            f"Negation flip: '{entity}' is negated in the findings but asserted "
            "as present in the impression."
        )
    if source is Polarity.ASSERTED and generated is Polarity.NEGATED:
        return (
            f"Negation flip: '{entity}' is present in the findings but negated "
            "in the impression."
        )
    if source is Polarity.UNCERTAIN and generated is Polarity.ASSERTED:
        return (
            f"Negation flip: '{entity}' is uncertain in the findings "
            "('cannot exclude') but asserted as present in the impression."
        )
    if source is Polarity.UNCERTAIN and generated is Polarity.NEGATED:
        return (
            f"Negation flip: '{entity}' is uncertain in the findings "
            "but negated in the impression."
        )
    return (
        f"Negation flip: '{entity}' polarity changed from {source.value} "
        f"to {generated.value}."
    )


def validate_negations(findings_text: str, impression_text: str) -> list[Flag]:
    """Flag entities whose polarity flips in the impression."""
    source_mentions = extract_entity_mentions(findings_text)
    generated_mentions = extract_entity_mentions(impression_text)
    source_polarity = entity_polarity(source_mentions)
    generated_polarity = entity_polarity(generated_mentions)

    flags: list[Flag] = []
    for entity, generated in generated_polarity.items():
        source = source_polarity.get(entity)
        if source is None or source is generated:
            continue
        if source is Polarity.ASSERTED and generated is Polarity.UNCERTAIN:
            continue
        flags.append(
            Flag(
                category=FlagCategory.NEGATION_FLIP,
                severity=_flip_severity(source, generated),
                source_evidence=_evidence(source_mentions, entity, source),
                generated_evidence=_evidence(generated_mentions, entity, generated),
                explanation=_flip_explanation(entity, source, generated),
            )
        )
    return flags

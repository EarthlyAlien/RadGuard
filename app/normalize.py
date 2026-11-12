"""Unicode and whitespace normalization for the deterministic pipeline."""

from __future__ import annotations

import re
import unicodedata

_SPACE_RE = re.compile(r"[ \t\u00a0\u202f\u2007\u2009\u200a\ufeff]+")
_DASHES = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00d7": "x",
        "\u2715": "x",
    }
)


def normalize_text(text: str) -> str:
    """NFKC-normalize, map unicode dashes/times, and collapse horizontal whitespace.

    Never raises: non-strings are coerced, NUL bytes are stripped.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\x00", "")
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_DASHES)
    text = _SPACE_RE.sub(" ", text)
    return text

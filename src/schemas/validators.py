"""
src/schemas/validators.py
=========================
Reusable, pure-function validators and Pydantic v2 field-validator helpers
shared across every input and output schema in this package.

Design principles
-----------------
* All helpers are stateless module-level callables — zero overhead from
  class instantiation on hot paths.
* Every helper that operates on text is Unicode-safe (works for Arabic
  and mixed Arabic/English).
* Helpers are intentionally small and composable; complex logic lives
  in the schema model_validators, not here.

Public API
----------
sanitize_medical_text    Strip leading/trailing whitespace; collapse internal
                         runs of whitespace; reject pure-whitespace values.
coerce_to_str            Turn None / list / int / ... into a clean str.
coerce_to_str_list       Turn None / str / dict / list into List[str].
normalize_patient_id     Uppercase and strip a patient code string.
clamp_float              Clamp an optional float to [lo, hi]; None passes through.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, List, Optional


# ── Constants ─────────────────────────────────────────────────────────────────

# Matches one or more consecutive whitespace characters (including Arabic ZWSP,
# ZWNBSP, RTL marks, etc.) after Unicode normalisation.
_WS_RUN = re.compile(r"\s+")

# Allowed pattern for patient IDs: letters, digits, and hyphens only.
# Mirrors the Field(pattern=...) constraint in AnalyzeInput but can be reused
# independently for custom validators.
PATIENT_ID_RE = re.compile(r"^[A-Za-z0-9\-]+$")

# Maximum lengths — single source of truth used by both schemas and tests.
MAX_TEXT_LEN: int = 100_000     # AnalyzeInput.text
MAX_FIELD_TEXT_LEN: int = 10_000  # RefineFieldInput.text
MAX_PATIENT_ID_LEN: int = 50
MAX_SESSION_ID_LEN: int = 128
MAX_NAME_LEN: int = 200


# ── Text helpers ──────────────────────────────────────────────────────────────

def sanitize_medical_text(value: str, *, field_name: str = "text") -> str:
    """Normalize Unicode, collapse whitespace, strip edges.

    Args:
        value:      Raw string from the request.
        field_name: Used only in the error message so callers get context.

    Returns:
        Cleaned string.

    Raises:
        ValueError: If ``value`` is empty or contains only whitespace after
                    normalization.
    """
    # NFC normalization keeps Arabic composed correctly and collapses ligatures.
    normalized = unicodedata.normalize("NFC", value)
    # Collapse runs of any whitespace (tabs, newlines, NBSP, …) to single space.
    collapsed = _WS_RUN.sub(" ", normalized).strip()
    if not collapsed:
        raise ValueError(f"'{field_name}' must contain non-whitespace characters")
    return collapsed


def coerce_to_str(value: Any, *, none_value: str = "") -> str:
    """Coerce any scalar/collection to a clean string.

    * ``None``              → ``none_value`` (default ``""``)
    * ``list`` / ``tuple``  → comma-joined (``"a, b, c"``)
    * ``dict``              → comma-joined values
    * anything else         → ``str(value).strip()``

    This mirrors the coercion logic already used in ``validation/schemas.py``
    so both layers behave identically.
    """
    if value is None:
        return none_value
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item).strip() for item in value if item is not None)
    if isinstance(value, dict):
        return ", ".join(str(v).strip() for v in value.values() if v is not None)
    return str(value).strip()


def coerce_to_str_list(value: Any) -> List[str]:
    """Coerce any value to a non-null ``List[str]`` with whitespace-trimmed elements.

    * ``None``   → ``[]``
    * ``str``    → single-element list
    * ``dict``   → list of values
    * ``list``   → cleaned element-by-element
    """
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, dict):
        value = list(value.values())
    return [str(item).strip() for item in value if item is not None and str(item).strip()]


def normalize_patient_id(value: Optional[str]) -> Optional[str]:
    """Strip and uppercase a patient code; return ``None`` for blank strings."""
    if value is None:
        return None
    cleaned = value.strip().upper()
    return cleaned if cleaned else None


def clamp_float(
    value: Optional[float],
    *,
    lo: float = 0.0,
    hi: float = float("inf"),
) -> Optional[float]:
    """Clamp a float to ``[lo, hi]``; pass ``None`` through unchanged."""
    if value is None:
        return None
    return max(lo, min(hi, value))


# ── Guard helpers (for use in model_validators) ───────────────────────────────

def require_non_blank(value: Optional[str], field_name: str = "field") -> str:
    """Assert the string is non-empty after stripping; raise ``ValueError`` otherwise."""
    if not (value or "").strip():
        raise ValueError(f"'{field_name}' must not be blank")
    return value.strip()  # type: ignore[return-value]

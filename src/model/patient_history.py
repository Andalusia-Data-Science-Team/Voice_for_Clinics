"""
Looks up a patient's past visits (roughly the last year) LIVE from the
hospital's HCMS replica database and answers two questions:

  1. get_patient_visits(patient_code, business_unit) — for the "Show Patient
     Previous Visits Over the Last Year" button: one dict per visit, newest
     first.

  2. format_history_for_prompt(visits) — a compact text block handed to the
     extraction prompt when the doctor opts in to "Use patient history?",
     so the LLM can factor prior diagnoses into today's ICD-10 codes /
     Final Diagnosis recommendation (see utils/prompt.py).

Matching is on PatientCode — the same "Code" value typed into the frontend's
Patient Info card — NOT the internal PatientID.

Each business unit (AHJ / AMH / ANC) lives in its own database, so the caller
must say which one to query. Server/database per unit come from
Config.BUSINESS_UNITS; the shared login comes from Config.PATIENT_DB_UID /
PATIENT_DB_PWD (i.e. the environment, never hardcoded).

The SQL lives in model/sql/patient_history.sql and takes ONE bound parameter
(the patient code). Every call opens a fresh connection and closes it, so
there's no stale state and no cache to invalidate. pyodbc is blocking —
call get_patient_visits from async code via asyncio.to_thread.

The return shape of get_patient_visits is identical to the old CSV-backed
version, so format_history_for_prompt / check_history_usage and the frontend
table are unchanged.
"""

import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyodbc

from core.config import Config
from core.telemetry import track_db_call

logger = logging.getLogger(__name__)

_SQL_PATH = Path(__file__).resolve().parent.parent / "sql" / "patient_history.sql"
_SQL = _SQL_PATH.read_text(encoding="utf-8")


class BusinessUnit(str, Enum):
    AHJ = "AHJ"
    AMH = "AMH"
    ANC = "ANC"


class PatientHistoryError(Exception):
    """
    The history lookup itself failed (DB unreachable, bad credentials, query
    error, missing config). Deliberately distinct from "this patient has no
    history on file", which is a normal empty result, not an error.
    """


def _build_connection_string(bu_cfg: dict) -> str:
    """Build ODBC connection string for HCMS database with Managed Identity or classic UID/PWD."""
    driver = Config.PATIENT_DB_DRIVER
    use_mi = Config.USE_MANAGED_IDENTITY_SQL

    if use_mi:
        # Managed Identity (Azure VM / App Service / Container Apps system-assigned identity)
        logger.info("[PatientHistory] Connecting to %s using Managed Identity (ActiveDirectoryMsi)", bu_cfg.get("database"))
        return (
            f"DRIVER={driver};"
            f"SERVER={bu_cfg['server']};"
            f"DATABASE={bu_cfg['database']};"
            "Authentication=ActiveDirectoryMsi;"
            "Encrypt=yes;"
            "TrustServerCertificate=no;"
        )

    # Classic UID / PWD
    uid = Config.patient_db_uid()
    pwd = Config.patient_db_pwd()
    if not uid or not pwd:
        raise PatientHistoryError("Patient history database credentials are not configured")

    pwd_escaped = pwd.replace("}", "}}")  # ODBC brace-escaping
    return (
        f"DRIVER={driver};"
        f"SERVER={bu_cfg['server']};"
        f"DATABASE={bu_cfg['database']};"
        f"UID={uid};"
        f"PWD={{{pwd_escaped}}};"
        "TrustServerCertificate=yes;"
    )


def _connect(business_unit: BusinessUnit) -> pyodbc.Connection:
    unit_cfg = Config.BUSINESS_UNITS.get(business_unit.value)
    if not unit_cfg:
        raise PatientHistoryError(f"No database configured for business unit {business_unit.value}")

    conn_str = _build_connection_string(unit_cfg)
    conn = pyodbc.connect(conn_str, timeout=10)  # login timeout (seconds)
    conn.timeout = 30                            # per-query timeout (seconds)
    return conn


def _fmt_date(value) -> str:
    return value.strftime("%Y-%m-%d") if hasattr(value, "strftime") else str(value or "")


_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def validate_history_citations(
    diagnosis: Dict[str, Any],
    visits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Verify claimed history dates against supplied visits without inventing dates."""
    result = dict(diagnosis)
    if str(result.get("history_considered", "")).strip().lower() != "yes":
        result["history_citation_missing"] = False
        result["history_cited_visit_dates"] = []
        return result

    visit_dates = {
        match.group(0)
        for visit in visits
        if (match := _ISO_DATE_RE.search(_fmt_date(visit.get("visit_date"))))
    }
    notes = str(result.get("history_notes") or "")
    reasoning = str(result.get("reasoning") or "")
    cited_dates = _ISO_DATE_RE.findall(notes)
    verified_dates = list(dict.fromkeys(date for date in cited_dates if date in visit_dates))
    has_invalid_date = any(date not in visit_dates for date in cited_dates)
    reasoning_missing_date = any(date not in reasoning for date in verified_dates)

    result["history_cited_visit_dates"] = verified_dates
    result["history_citation_missing"] = (
        not verified_dates or has_invalid_date or reasoning_missing_date
    )
    return result


def _split_diagnoses(raw: Optional[str]) -> List[str]:
    if not raw or raw.strip().upper() == "NULL":
        return []
    return [d.strip() for d in raw.split("|") if d.strip()]


def get_patient_visits(patient_code: str, business_unit: BusinessUnit) -> List[Dict]:
    """
    Returns every visit in the last 12 months for this PatientCode in the
    given business unit, newest first (the SQL already orders them). Each item:
      {visit_id, visit_date, specialty, icd10_diagnoses (list[str]),
       department, visit_type, previous_visit_count}
    Empty list if the patient has no history on file (not an error — plenty of
    patients will be genuinely new).

    Raises PatientHistoryError if the lookup itself fails.
    """
    code = str(patient_code).strip()
    if not code:
        return []

    try:
        with track_db_call(business_unit=business_unit.value):
            conn = _connect(business_unit)
            try:
                cursor = conn.cursor()
                cursor.execute(_SQL, code)
                columns = [c[0] for c in cursor.description]
                rows = [dict(zip(columns, r)) for r in cursor.fetchall()]
            finally:
                conn.close()
    except PatientHistoryError:
        raise
    except pyodbc.Error as e:
        logger.error(
            "[PatientHistory] %s lookup failed for patient_code=%r: %s",
            business_unit.value, code, e,
        )
        raise PatientHistoryError(
            f"Could not retrieve patient history from {business_unit.value}"
        ) from e

    logger.info(
        "[PatientHistory] %s: %d visit(s) found for patient_code=%r",
        business_unit.value, len(rows), code,
    )

    return [
        {
            "visit_id":             str(r.get("VisitID") or ""),
            "visit_date":           _fmt_date(r.get("VisitDate")),
            "specialty":            (r.get("Specialty") or "").strip(),
            "icd10_diagnoses":      _split_diagnoses(r.get("CurrentICD10")),
            "department":           (r.get("Department") or "").strip(),
            "visit_type":           (r.get("VisitTypeName") or "").strip(),
            "previous_visit_count": str(r.get("PreviousVisitCount") or 0),
        }
        for r in rows
    ]


def format_history_for_prompt(visits: List[Dict]) -> str:
    """
    Compact, LLM-friendly text block — one line per past visit. Returns ""
    if there's nothing to show, so callers can skip the history section of
    the prompt entirely rather than sending an empty block.
    """
    if not visits:
        return ""

    lines = []
    for v in visits:
        dx = "; ".join(v["icd10_diagnoses"]) if v["icd10_diagnoses"] else "No diagnosis recorded"
        lines.append(
            f"- {v['visit_date']} | {v['specialty']} | {v['department']} ({v['visit_type']}): {dx}"
        )
    return "\n".join(lines)


def _code_family(code_entry: Optional[str]) -> str:
    """
    Pulls the ICD-10 "family"/category out of either a bare code ("E11.65")
    or a "code - description" string ("E11.65 - Type 2 diabetes mellitus...").
    Returns e.g. "E11" for "E11.65", "J45" for "J45 - Asthma". Used only for
    the soft cross-check below — categories, not full codes, because a
    genuinely-relevant history match is still useful even when the specific
    subcode differs (e.g. E11.65 today vs E11.9 previously).
    """
    if not code_entry:
        return ""
    code_part = code_entry.split(" - ", 1)[0].strip()
    code_part = code_part.split(".")[0]
    return code_part.upper()


def check_history_usage(visits: List[Dict], extracted_features: Dict) -> Dict:
    """
    Soft cross-check between what the model SAID it did with patient history
    (`history_considered` / `history_notes` in extracted_features) and what
    actually shows up in `icd10_codes` / `final_diagnosis` — by comparing
    ICD-10 code families. This is a heuristic, not a hard validator: overlap
    is a reasonable proxy for "history was used," but its absence doesn't
    prove it wasn't (the model can legitimately factor in a comorbidity
    without that producing a matching code), and overlap can occasionally be
    coincidental. Treat `flag` as "worth a second look," not "wrong."

    Returns:
      {
        "history_provided":   bool — was there any history to consider at all,
        "claimed_considered": bool — did the model say history_considered == "Yes",
        "overlap_found":      bool — any ICD-10 family shared between history and today's codes,
        "overlap_families":   [str] — which families overlapped,
        "flag": None | "claims_used_no_overlap" | "overlap_but_not_flagged",
      }
    """
    history_families = set()
    for v in (visits or []):
        for dx in v.get("icd10_diagnoses", []):
            fam = _code_family(dx)
            if fam:
                history_families.add(fam)

    if not history_families:
        return {
            "history_provided":   False,
            "claimed_considered": False,
            "overlap_found":      False,
            "overlap_families":   [],
            "flag":               None,
        }

    extracted_features = extracted_features or {}
    current_codes = list(extracted_features.get("icd10_codes") or [])
    final_dx      = extracted_features.get("final_diagnosis") or ""

    current_families = {_code_family(c) for c in current_codes if c}
    if final_dx:
        current_families.add(_code_family(final_dx))
    current_families.discard("")

    overlap = sorted(history_families & current_families)
    claimed = str(extracted_features.get("history_considered", "")).strip().lower() == "yes"

    flag = None
    if claimed and not overlap:
        flag = "claims_used_no_overlap"
    elif overlap and not claimed:
        flag = "overlap_but_not_flagged"

    return {
        "history_provided":   True,
        "claimed_considered": claimed,
        "overlap_found":      bool(overlap),
        "overlap_families":   overlap,
        "flag":               flag,
    }
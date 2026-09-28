"""
Loads "Visit Level History.csv" (PatientCode -> past visits, over roughly
the last year) and answers two questions:

  1. get_patient_visits(patient_code) — for the "Show Patient Previous Visits
     Over the Last Year" button: the raw rows for that patient, newest first.

  2. format_history_for_prompt(visits) — a compact text block handed to the
     extraction prompt when the doctor opts in to "Use patient history?",
     so the LLM can factor prior diagnoses into today's ICD-10 codes /
     Final Diagnosis recommendation (see utils/prompt.py).

Matching is on the sheet's PatientCode column — the same "Code" value typed
into the frontend's Patient Info card — NOT the PatientID column. The two
are different identifiers in this sheet (PatientID is the internal record
ID; PatientCode is the human-facing code the doctor actually has on hand),
and only PatientCode is something the doctor can type in. Older CSVs that
only have a PatientID column (no PatientCode) are still supported as a
fallback, so this doesn't break on a sheet from before the two were split.

The file is read fresh on every call (it's tiny — a few thousand rows at
most for a single clinic) rather than cached, so updates to the CSV show up
immediately without a restart. Delimiter (tab or comma) is auto-detected so
either a straight tab-paste or a proper comma CSV both work.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

from core.config import Config

logger = logging.getLogger(__name__)

DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "Visit Level History.csv"


def _csv_path() -> Path:
    configured = getattr(Config, "PATIENT_HISTORY_CSV_PATH", None)
    return Path(configured) if configured else DEFAULT_CSV_PATH


def _split_diagnoses(raw: Optional[str]) -> List[str]:
    if not raw or raw.strip().upper() == "NULL":
        return []
    return [d.strip() for d in raw.split("|") if d.strip()]


def _row_patient_code(row: Dict[str, str]) -> str:
    """
    The sheet's PatientCode column is the match key (same value as the
    frontend's Patient Info "Code" field). Falls back to PatientID only for
    older CSVs that predate the PatientCode column existing at all — if a
    row has a PatientCode column, that value always wins even when empty,
    so a genuinely blank code doesn't silently fall back to matching on the
    unrelated internal PatientID.
    """
    if "PatientCode" in row:
        return (row.get("PatientCode") or "").strip()
    return (row.get("PatientID") or "").strip()


def _row_current_icd10(row: Dict[str, str]) -> Optional[str]:
    """CurrentICD10 is the sheet's present column name; ICD10_Diagnoses is
    kept as a fallback for older CSVs."""
    if "CurrentICD10" in row:
        return row.get("CurrentICD10")
    return row.get("ICD10_Diagnoses")


def _sniff_dialect(sample: str):
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,")
    except csv.Error:
        # Sniffer gives up on very short/uniform samples — fall back to
        # whichever separator actually shows up in the header line.
        first_line = sample.splitlines()[0] if sample else ""
        return csv.excel_tab if "\t" in first_line else csv.excel


def _load_rows() -> List[Dict[str, str]]:
    path = _csv_path()
    if not path.exists():
        logger.warning(f"[PatientHistory] CSV not found at {path}")
        return []
    try:
        with path.open(newline="", encoding="cp1252") as f:
            sample = f.read(4096)
            f.seek(0)
            dialect = _sniff_dialect(sample)
            reader = csv.DictReader(f, dialect=dialect)
            return [row for row in reader]
    except Exception as e:
        logger.error(f"[PatientHistory] Failed to read {path}: {e}")
        return []


def _visit_sort_key(visit: Dict) -> tuple:
    # VisitDate is stored as "M/D/YYYY" text — parse for sorting, newest first.
    try:
        m, d, y = visit["visit_date"].split("/")
        return (int(y), int(m), int(d))
    except Exception:
        return (0, 0, 0)


def get_patient_visits(patient_code: str) -> List[Dict]:
    """
    Returns every row in the CSV matching this PatientCode (the same value
    the doctor types into the "Code" field on the Patient Info card),
    newest visit first. Each item:
      {visit_id, visit_date, specialty, icd10_diagnoses (list[str]),
       department, visit_type, previous_visit_count}
    Empty list if the patient has no history on file (not an error —
    plenty of patients will be genuinely new).
    """
    code = str(patient_code).strip()
    if not code:
        return []

    visits = []
    for row in _load_rows():
        if _row_patient_code(row) != code:
            continue
        visits.append({
            "visit_id":             (row.get("VisitID") or "").strip(),
            "visit_date":           (row.get("VisitDate") or "").strip(),
            "specialty":            (row.get("Specialty") or "").strip(),
            "icd10_diagnoses":      _split_diagnoses(_row_current_icd10(row)),
            "department":           (row.get("Department") or "").strip(),
            "visit_type":           (row.get("VisitTypeName") or "").strip(),
            "previous_visit_count": (row.get("PreviousVisitCount") or "").strip(),
        })

    visits.sort(key=_visit_sort_key, reverse=True)
    return visits


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
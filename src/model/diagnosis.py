import time
import logging
from typing import Optional
from model.llm_service import LLMService
from core.config import Config

logger = logging.getLogger(__name__)


def _as_str_list(value) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, dict):
        value = list(value.values())
    return [str(v).strip() for v in value if str(v).strip()]


class DiagnosisRecommender:
    """Step 2 — ICD-10 recommendation + reasoning (ER also: differential
    diagnoses + triage priority). Runs in parallel with ExtractFeature; the only
    step that receives patient history."""

    @staticmethod
    def recommend(
        end_text: str,
        department: str = "opd",
        patient_history_text: str = "",
        is_conversation: bool = False,
        model: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> tuple[dict, str]:
        """Returns (json_data, summary). json_data keys:
        icd10_codes (list[str], most likely first), reasoning (str),
        history_considered / history_notes (str), and for ER only:
        differential_diagnoses (list[str]), triage_priority (str)."""
        if not end_text or not isinstance(end_text, str):
            raise ValueError("Input end_text must be a non-empty string")

        dept = (department or "").strip().lower()
        start = time.time()
        try:
            logger.info(f"[Diagnosis] {dept} recommendation, model={model or 'default'}, history={'yes' if patient_history_text else 'no'}")
            out = LLMService.recommend_diagnosis(
                translated_text=end_text,
                api_key=Config.ai_foundry_key(),
                department=dept,
                patient_history_text=patient_history_text,
                is_conversation=is_conversation,
                model_id=model,
                meta=meta,
            )
            data = dict(out.get("json_data", {})) if out else {}
            summary = out.get("summary", "") if out else ""

            # Normalize — models differ in how strictly they follow the list/string shapes.
            data["icd10_codes"] = _as_str_list(data.get("icd10_codes"))
            data["reasoning"] = str(data.get("reasoning") or "").strip()
            data["history_considered"] = str(data.get("history_considered") or "No").strip()
            data["history_notes"] = str(data.get("history_notes") or "").strip()
            if dept == "er":
                data["differential_diagnoses"] = _as_str_list(data.get("differential_diagnoses"))
                data["triage_priority"] = str(data.get("triage_priority") or "").strip()
            else:
                data.pop("differential_diagnoses", None)
                data.pop("triage_priority", None)

            if patient_history_text and "history_considered" not in (out or {}).get("json_data", {}):
                logger.warning("[Diagnosis] History was supplied but the response had no 'history_considered' key.")
            logger.info(f"[Diagnosis] completed in {time.time() - start:.2f}s")
            return data, summary
        except Exception as e:
            logger.error(f"[Diagnosis] Failed: {e}")
            raise
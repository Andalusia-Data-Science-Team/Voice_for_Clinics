import time
import logging
from typing import Optional
from model.llm_service import LLMService
from core.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtractFeature:
    """Step 1 — clinical-note FIELD extraction (no diagnoses, no patient history)."""

    @staticmethod
    def extract(
        end_text: str,
        department: str = "opd",
        is_conversation: bool = False,
        model: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> tuple[dict, str]:
        """
        Args:
            end_text: translated/final text to process.
            department: "er", "opd" or "dental" (unknown -> "opd").
            is_conversation: doctor-patient conversation format.
            model: id from core.model_registry.EXTRACTION_MODELS (None -> default).
            meta: optional dict that receives latency / tokens / cost of the LLM call.

        Returns: (json_data, reasoning_summary)
        """
        if not end_text or not isinstance(end_text, str):
            raise ValueError("Input end_text must be a non-empty string")

        start = time.time()
        try:
            logger.info(f"[ExtractFeature] {department} extraction, model={model or 'default'}, conversation={is_conversation}")
            out = LLMService.extract_features(
                translated_text=end_text,
                api_key=Config.OPENROUTER_API_KEY,
                department=department,
                is_conversation=is_conversation,
                model_id=model,
                meta=meta,
            )
            json_data = out.get("json_data", {}) if out else {}
            summary = out.get("summary", "") if out else ""
            logger.info(f"[ExtractFeature] {department} extraction completed in {time.time() - start:.2f}s")
            return json_data, summary
        except Exception as e:
            logger.error(f"[ExtractFeature] Extraction failed: {e}")
            raise
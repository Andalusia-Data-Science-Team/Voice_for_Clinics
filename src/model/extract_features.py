import time
import logging
from model.llm_service import LLMService
from core.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtractFeature:
    """Module for extracting structured features from text using LLMService."""
    @staticmethod
    def extract(
        end_text: str,
        department: str = "opd",
        patient_history_text: str = "",
        is_conversation: bool = False,
    ) -> tuple[dict, str]:
        """
        Extract features from final translated text using LLM.
        Features are defined per-department in the prompt templates, not
        passed by the caller.

        Args:
            end_text (str): The translated text to process.
            department (str): "er", "opd", or "dental" — selects which
                clinical note schema/prompt to use. Defaults to "opd".
            patient_history_text (str): Pre-formatted patient visit-history
                text (see model.patient_history.format_history_for_prompt),
                or "" if the doctor opted out or no history exists. Only
                influences icd10_codes / final_diagnosis.
            is_conversation (bool): If True, extracts from doctor-patient conversation format.

        Returns:
            tuple: (json_data: dict, reasoning: str)
        """
        if not end_text or not isinstance(end_text, str):
            raise ValueError("Input end_text must be a non-empty string")

        extraction_start = time.time()
        try:
            mode_label = "conversation" if is_conversation else "single-speaker"
            logger.info(
                f"[ExtractFeature] Starting {mode_label} {department} feature extraction "
                f"(patient_history={'yes' if patient_history_text else 'no'})"
            )

            features_output = LLMService.extract_features(
                translated_text=end_text,
                api_key=Config.OPENROUTER_API_KEY,
                department=department,
                patient_history_text=patient_history_text,
                is_conversation=is_conversation,
            )

            json_data = features_output.get("json_data", {}) if features_output else {}
            reasoning = features_output.get("summary", "") if features_output else ""

            extraction_time = time.time() - extraction_start
            logger.info(f"[ExtractFeature] {mode_label.capitalize()} {department} extraction completed in {extraction_time:.2f}s")
            logger.debug(f"[ExtractFeature] Extracted features: {json_data}")
            logger.debug(f"[ExtractFeature] Reasoning: {reasoning[:200]}...")

            if patient_history_text:
                # Verification signal for point "make sure the LLM actually uses
                # the patient history" — log whether the model actually reported
                # using it, so a doctor/engineer can audit this without guessing.
                considered = json_data.get("history_considered", "<missing>")
                logger.info(f"[ExtractFeature] Patient history was supplied — model reports history_considered={considered!r}")
                if considered == "<missing>":
                    logger.warning(
                        "[ExtractFeature] Patient history was supplied but the model's response had no "
                        "'history_considered' key — it may not have registered the instruction."
                    )

            return json_data, reasoning

        except Exception as e:
            logger.error(f"[ExtractFeature] Extraction failed: {str(e)}")
            raise
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
        is_conversation: bool = False,
    ) -> tuple[dict, str]:
        """
        Extract features from final translated text using LLM.
        Features are defined in the prompt template, not passed by the caller.

        Args:
            end_text (str): The translated text to process.
            is_conversation (bool): If True, extracts from doctor-patient conversation format.

        Returns:
            tuple: (json_data: dict, reasoning: str)
        """
        if not end_text or not isinstance(end_text, str):
            raise ValueError("Input end_text must be a non-empty string")

        extraction_start = time.time()
        try:
            mode_label = "conversation" if is_conversation else "single-speaker"
            logger.info(f"[ExtractFeature] Starting {mode_label} feature extraction")

            features_output = LLMService.extract_features(
                translated_text=end_text,
                api_key=Config.EXTRACTION_API_KEY,
                is_conversation=is_conversation,
            )

            json_data = features_output.get("json_data", {})
            reasoning = features_output.get("reasoning", "")

            extraction_time = time.time() - extraction_start
            logger.info(f"[ExtractFeature] {mode_label.capitalize()} extraction completed in {extraction_time:.2f}s")
            logger.debug(f"[ExtractFeature] Extracted features: {json_data}")
            logger.debug(f"[ExtractFeature] Reasoning: {reasoning[:200]}...")

            if is_conversation and "conversation_summary" in json_data:
                logger.info(f"[ExtractFeature] Conversation summary extracted: {len(json_data['conversation_summary'])} chars")

            return json_data, reasoning

        except Exception as e:
            logger.error(f"[ExtractFeature] Extraction failed: {str(e)}")
            raise
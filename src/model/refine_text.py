import time
import logging
from model.llm_service import LLMService
from core.config import Config

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RefineText:
    """Module for refining transcribed text using LLMService."""

    @staticmethod
    def refining_transcription(raw_text: str, language: str, is_conversation: bool = False) -> str:
        """
        Refine a transcribed text depending on the language and mode.

        Args:
            raw_text (str): The raw transcribed text.
            language (str): "ar" for Arabic, "en" for English, others fallback to English.
            is_conversation (bool): If True, treats input as doctor-patient conversation.

        Returns:
            str: Refined text (with speaker labels if conversation mode).
        """
        if not raw_text or not isinstance(raw_text, str):
            raise ValueError("Input raw_text must be a non-empty string")

        refine_start = time.time()
        try:
            mode_label = "conversation" if is_conversation else "single-speaker"
            logger.info(f"[RefineText] Starting {mode_label} refinement for language: {language}")
            
            if language.lower() == "ar":
                refined_text = LLMService.refine_ar_transcription(
                    raw_text,
                    Config.REFINE_API_KEY,
                    is_conversation=is_conversation
                )
            else:
                refined_text = LLMService.refine_en_transcription(
                    raw_text,
                    Config.REFINE_API_KEY,
                    is_conversation=is_conversation
                )

            refine_time = time.time() - refine_start
            logger.info(f"[RefineText] {mode_label.capitalize()} refinement completed in {refine_time:.2f}s")
            logger.debug(f"[RefineText] Output: {refined_text[:200]}...")
            
            # Log if speaker labels were detected (for conversation mode validation)
            if is_conversation:
                has_labels = "**DOCTOR:**" in refined_text or "**PATIENT:**" in refined_text
                logger.info(f"[RefineText] Speaker labels detected: {has_labels}")

            return refined_text

        except Exception as e:
            logger.error(f"[RefineText] Refinement failed: {str(e)}")
            raise
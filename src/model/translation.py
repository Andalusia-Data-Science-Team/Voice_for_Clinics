import time
import logging
from model.llm_service import LLMService
from core.config import Config

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Translate:
    """Module for translating refined text into English using LLMService."""

    @staticmethod
    def translate(refined_text: str, is_conversation: bool = False) -> str:
        """
        Translate refined text into English.

        Args:
            refined_text (str): The text after refinement.
            is_conversation (bool): If True, preserves conversation structure with speaker labels.

        Returns:
            str: Translated English text (with speaker labels if conversation mode).
        """
        if not refined_text or not isinstance(refined_text, str):
            raise ValueError("Input refined_text must be a non-empty string")

        translation_start = time.time()
        try:
            mode_label = "conversation" if is_conversation else "single-speaker"
            logger.info(f"[Translate] Starting {mode_label} translation")
            
            translated_text = LLMService.translate_to_eng(
                refined_text,
                Config.TRANSLATE_API_KEY,
                is_conversation=is_conversation
            )

            translation_time = time.time() - translation_start
            logger.info(f"[Translate] {mode_label.capitalize()} translation completed in {translation_time:.2f}s")
            logger.debug(f"[Translate] Output: {translated_text[:200]}...")
            
            # Validate speaker labels preservation in conversation mode
            if is_conversation:
                has_labels = "**DOCTOR:**" in translated_text or "**PATIENT:**" in translated_text
                logger.info(f"[Translate] Speaker labels preserved: {has_labels}")
                if not has_labels:
                    logger.warning("[Translate] Warning: Expected speaker labels not found in conversation mode")

            return translated_text

        except Exception as e:
            logger.error(f"[Translate] Translation failed: {str(e)}")
            raise
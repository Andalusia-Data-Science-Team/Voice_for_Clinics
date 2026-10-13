import time
import logging
from typing import Optional

from model.llm_service import LLMService
from model.specialty_terms import correct_specialty_terms
from core.config import Config

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RefineText:
    """Module for refining transcribed text using LLMService."""

    @staticmethod
    def refining_transcription(
        raw_text: str,
        language: str,
        is_conversation: bool = False,
        specialty: Optional[str] = None,
    ) -> str:
        """
        Refine a transcribed text depending on the language, mode, and specialty.

        Args:
            raw_text (str): The raw transcribed text.
            language (str): "ar" for Arabic, "en" for English, others fallback to English.
            is_conversation (bool): If True, treats input as doctor-patient conversation.
            specialty (str | None): One of the fixed specialty names (see
                model.specialty_terms.SPECIALTIES). Used two ways: (1) a fast,
                deterministic fuzzy-matching pass corrects near-miss specialty
                terms before AND after the LLM call, and (2) the LLM refinement
                prompt is told which specialty it's in. None/unknown skips both.

        Returns:
            str: Refined text (with speaker labels if conversation mode).
        """
        if not raw_text or not isinstance(raw_text, str):
            raise ValueError("Input raw_text must be a non-empty string")

        refine_start = time.time()
        try:
            mode_label = "conversation" if is_conversation else "single-speaker"
            logger.info(
                f"[RefineText] Starting {mode_label} refinement for language: {language}, specialty: {specialty or 'none'}"
            )

            # Deterministic pre-pass: fix obvious ASR near-misses of known
            # specialty terms before the LLM ever sees the text.
            pre_corrected = correct_specialty_terms(raw_text, specialty)

            if language.lower() == "ar":
                refined_text = LLMService.refine_ar_transcription(
                    pre_corrected,
                    Config.AI_FOUNDRY_KEY,
                    is_conversation=is_conversation,
                    specialty=specialty,
                )
            else:
                refined_text = LLMService.refine_en_transcription(
                    pre_corrected,
                    Config.AI_FOUNDRY_KEY,
                    is_conversation=is_conversation,
                    specialty=specialty,
                )

            # Second pass on the LLM's output — a safety net in case the LLM
            # introduced or left a near-miss spelling of a glossary term.
            refined_text = correct_specialty_terms(refined_text, specialty)

            refine_time = time.time() - refine_start
            logger.info(f"[RefineText] {mode_label.capitalize()} refinement completed in {refine_time:.2f}s")
            logger.debug(f"[RefineText] Output: {refined_text[:200]}...")
            
            # Log if speaker labels were detected (for conversation mode validation)
            if is_conversation:
                has_labels = any(
                    label in refined_text
                    for label in ("Doctor:", "Patient:", "الدكتور:", "المريض:")
                )
                logger.info(f"[RefineText] Speaker labels detected: {has_labels}")

            return refined_text

        except Exception as e:
            logger.error(f"[RefineText] Refinement failed: {str(e)}")
            raise
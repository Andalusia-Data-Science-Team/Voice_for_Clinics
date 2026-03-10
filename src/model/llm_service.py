import fireworks.client
import logging
from typing import Optional, Type
from pydantic import BaseModel, ValidationError

from utils import prompt as prompt_utils
from utils.utils import safe_parse_json

# ---------------- Logger ---------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------- Pydantic Models ---------------- #

class ExtractedFeatures(BaseModel):
    json_data: dict
    reasoning: str

class QuestionAnswer(BaseModel):
    question: str
    answer: Optional[str]
    needs_asking: bool
    category: str

class GeneratedQuestions(BaseModel):
    questions: list[QuestionAnswer]
    reasoning: str


# ---------------- LLM Service ---------------- #

class LLMService:
    """Service wrapper around Fireworks LLM API for refinement, translation, and question generation."""

    @staticmethod
    def refine_en_transcription(raw_text: str, api_key: str, is_conversation: bool = False):
        return LLMService.process_text(
            text=raw_text, api_key=api_key, model="deepseek",
            prompt_type="refine_english", is_conversation=is_conversation
        )

    @staticmethod
    def refine_ar_transcription(raw_text: str, api_key: str, is_conversation: bool = False):
        return LLMService.process_text(
            text=raw_text, api_key=api_key, model="deepseek",
            prompt_type="refine_arabic", is_conversation=is_conversation
        )

    @staticmethod
    def translate_to_eng(refined_text: str, api_key: str, is_conversation: bool = False):
        return LLMService.process_text(
            text=refined_text, api_key=api_key, model="deepseek",
            prompt_type="translate", is_conversation=is_conversation
        )

    @staticmethod
    def extract_features(translated_text: str, api_key: str, is_conversation: bool = False):
        # features are now hardcoded in the prompt template — no longer passed here
        return LLMService.process_text(
            text=translated_text, api_key=api_key, model="llama",
            prompt_type="extract_dynamic",
            pydantic_model=ExtractedFeatures,
            is_conversation=is_conversation,
        )

    @staticmethod
    def generate_questions(translated_text: str, api_key: str, is_conversation: bool = False):
        return LLMService.process_text(
            text=translated_text, api_key=api_key, model="llama",
            prompt_type="generate_questions",
            pydantic_model=GeneratedQuestions,
            is_conversation=is_conversation,
        )

    @staticmethod
    def process_text(
        text: str,
        api_key: str,
        model: str,
        prompt_type: str,
        features: Optional[list] = None,
        pydantic_model: Optional[Type[BaseModel]] = None,
        is_conversation: bool = False,
    ):
        fireworks.client.api_key = api_key
        model_account = "accounts/fireworks/models/deepseek-v3p1"

        prompt = LLMService._get_prompt(prompt_type, text, features, is_conversation)
        logger.debug(f"Generated prompt (is_conversation={is_conversation}): {prompt[:200]}...")

        result = LLMService._call_llm_api(
            model_account=model_account,
            prompt=prompt,
            pydantic_model=pydantic_model,
        )

        return result if result else text

    @staticmethod
    def _get_prompt(prompt_type: str, text: str, features: Optional[list], is_conversation: bool):
        mapping = {
            ("refine_english", False): prompt_utils.get_refine_english_prompt_deepseek,
            ("refine_english", True): prompt_utils.get_refine_english_prompt_deepseek_conversation,
            ("refine_arabic", False): prompt_utils.get_refine_arabic_prompt_deepseek,
            ("refine_arabic", True): prompt_utils.get_refine_arabic_prompt_deepseek_conversation,
            ("translate", False): prompt_utils.get_translation_prompt_deepseek,
            ("translate", True): prompt_utils.get_translation_prompt_deepseek_conversation,
            # features arg is now unused — prompts are self-contained
            ("extract_dynamic", False): prompt_utils.get_dynamic_extraction_prompt_llama,
            ("extract_dynamic", True): prompt_utils.get_dynamic_extraction_prompt_llama_conversation,
            ("generate_questions", False): prompt_utils.get_question_generation_prompt_llama,
            ("generate_questions", True): prompt_utils.get_question_generation_prompt_llama_conversation,
        }

        key = (prompt_type, is_conversation)
        func = mapping.get(key)

        if func is None:
            raise ValueError(f"Unsupported prompt type={prompt_type} with is_conversation={is_conversation}")

        return func(text)

    @staticmethod
    def _call_llm_api(
        model_account: str,
        prompt: str,
        pydantic_model: Optional[Type[BaseModel]] = None,
        temperature: float = 0,
    ):
        params = {
            "model": model_account,
            "prompt": prompt,
            "max_tokens": 4096,
            "temperature": temperature,
        }

        if pydantic_model:
            params["response_format"] = {"type": "json_object", "schema": pydantic_model.model_json_schema()}

        response = fireworks.client.Completion.create(**params)

        if not response.choices or not response.choices[0].text.strip():
            logger.warning("LLM returned empty response")
            return None

        raw_output = response.choices[0].text.strip()

        if pydantic_model:
            try:
                parsed_output = safe_parse_json(raw_output)
                validated_output = pydantic_model(**parsed_output)
                return validated_output.model_dump()
            except (ValueError, ValidationError) as e:
                logger.error(f"Structured output validation failed: {e}")
                return None

        return raw_output
import asyncio
import logging
from typing import Optional, Type
from openai import OpenAI
from pydantic import BaseModel, ValidationError, field_validator

from utils import prompt as prompt_utils
from utils.utils import safe_parse_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------- Pydantic Models ---------------- #

class ExtractedFeatures(BaseModel):
    json_data: dict


class QuestionAnswer(BaseModel):
    question: str
    answer: Optional[str] = None
    needs_asking: bool
    category: str

    @field_validator("answer", mode="before")
    @classmethod
    def coerce_answer_to_str(cls, v):
        if v is None:
            return None
        return str(v) if not isinstance(v, str) else v


class GeneratedQuestions(BaseModel):
    questions: list[QuestionAnswer]


# ---------------- OpenRouter Client ---------------- #

def _get_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


# ---------------- LLM Service ---------------- #

class LLMService:

    DEEPSEEK_MODEL = "deepseek/deepseek-chat-v3-0324"   # fast DeepSeek for text tasks
    FAST_MODEL     = "openai/gpt-4o-mini"               # fast + reliable structured JSON

    # ── Non-streaming text methods ──────────────────────────────────────
    @staticmethod
    def refine_en_transcription(raw_text: str, api_key: str, is_conversation: bool = False):
        return LLMService._call_text(raw_text, api_key, "refine_english", is_conversation)

    @staticmethod
    def refine_ar_transcription(raw_text: str, api_key: str, is_conversation: bool = False):
        return LLMService._call_text(raw_text, api_key, "refine_arabic", is_conversation)

    @staticmethod
    def translate_to_eng(refined_text: str, api_key: str, is_conversation: bool = False):
        return LLMService._call_text(refined_text, api_key, "translate", is_conversation)

    # ── Streaming text methods ───────────────────────────────────────────
    @staticmethod
    async def refine_en_transcription_stream(raw_text: str, api_key: str, is_conversation: bool = False):
        async for delta in LLMService._stream_text(raw_text, api_key, "refine_english", is_conversation):
            yield delta

    @staticmethod
    async def refine_ar_transcription_stream(raw_text: str, api_key: str, is_conversation: bool = False):
        async for delta in LLMService._stream_text(raw_text, api_key, "refine_arabic", is_conversation):
            yield delta

    @staticmethod
    async def translate_to_eng_stream(refined_text: str, api_key: str, is_conversation: bool = False):
        async for delta in LLMService._stream_text(refined_text, api_key, "translate", is_conversation):
            yield delta

    # ── Structured output methods (extraction + questions) ───────────────
    @staticmethod
    def extract_features(translated_text: str, api_key: str, is_conversation: bool = False):
        return LLMService._call_structured(
            text=translated_text,
            api_key=api_key,
            prompt_type="extract_dynamic",
            pydantic_model=ExtractedFeatures,
            is_conversation=is_conversation,
        )

    @staticmethod
    def generate_questions(translated_text: str, api_key: str, is_conversation: bool = False):
        return LLMService._call_structured(
            text=translated_text,
            api_key=api_key,
            prompt_type="generate_questions",
            pydantic_model=GeneratedQuestions,
            is_conversation=is_conversation,
        )

    # ── Internal: plain text call (refine / translate) ───────────────────
    @staticmethod
    def _call_text(text: str, api_key: str, prompt_type: str, is_conversation: bool) -> str:
        prompt = LLMService._get_prompt(prompt_type, text, is_conversation)
        client = _get_client(api_key)
        response = client.chat.completions.create(
            model=LLMService.DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0,
        )
        return response.choices[0].message.content.strip() if response.choices else text

    # ── Internal: structured JSON call (extraction / questions) ──────────
    @staticmethod
    def _call_structured(
        text: str,
        api_key: str,
        prompt_type: str,
        pydantic_model: Type[BaseModel],
        is_conversation: bool,
    ):
        prompt = LLMService._get_prompt(prompt_type, text, is_conversation)
        client = _get_client(api_key)

        # Use json_object — broader support across OpenRouter models than json_schema
        response = client.chat.completions.create(
            model=LLMService.FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0,
            response_format={"type": "json_object"},
        )

        if not response.choices or not response.choices[0].message.content.strip():
            logger.warning("LLM returned empty response")
            return None

        raw_output = response.choices[0].message.content.strip()
        logger.debug(f"Raw structured output: {raw_output[:300]}")

        try:
            parsed = safe_parse_json(raw_output)

            # Unwrap single-key wrapper e.g. {"GeneratedQuestions": {...}}
            if isinstance(parsed, dict) and len(parsed) == 1:
                only_key = next(iter(parsed))
                inner = parsed[only_key]
                model_fields = set(pydantic_model.model_fields.keys())
                if isinstance(inner, dict) and model_fields & set(inner.keys()):
                    parsed = inner

            validated = pydantic_model(**parsed)
            return validated.model_dump()

        except (ValueError, ValidationError) as e:
            logger.error(f"Structured output validation failed: {e}")
            logger.error(f"Raw output was: {raw_output[:500]}")
            return None

    # ── Internal: streaming text ──────────────────────────────────────────
    @staticmethod
    async def _stream_text(
        text: str,
        api_key: str,
        prompt_type: str,
        is_conversation: bool = False,
    ):
        prompt = LLMService._get_prompt(prompt_type, text, is_conversation)
        client = _get_client(api_key)

        response = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model=LLMService.DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0,
                stream=True,
            )
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ── Prompt dispatch ───────────────────────────────────────────────────
    @staticmethod
    def _get_prompt(prompt_type: str, text: str, is_conversation: bool) -> str:
        mapping = {
            ("refine_english",     False): prompt_utils.get_refine_english_prompt_deepseek,
            ("refine_english",     True):  prompt_utils.get_refine_english_prompt_deepseek_conversation,
            ("refine_arabic",      False): prompt_utils.get_refine_arabic_prompt_deepseek,
            ("refine_arabic",      True):  prompt_utils.get_refine_arabic_prompt_deepseek_conversation,
            ("translate",          False): prompt_utils.get_translation_prompt_deepseek,
            ("translate",          True):  prompt_utils.get_translation_prompt_deepseek_conversation,
            ("extract_dynamic",    False): prompt_utils.get_dynamic_extraction_prompt_llama,
            ("extract_dynamic",    True):  prompt_utils.get_dynamic_extraction_prompt_llama_conversation,
            ("generate_questions", False): prompt_utils.get_question_generation_prompt_llama,
            ("generate_questions", True):  prompt_utils.get_question_generation_prompt_llama_conversation,
        }
        func = mapping.get((prompt_type, is_conversation))
        if func is None:
            raise ValueError(f"Unsupported prompt_type={prompt_type}, is_conversation={is_conversation}")
        return func(text)

    # ── Legacy process_text (kept for backward compat) ────────────────────
    @staticmethod
    def process_text(text, api_key, model, prompt_type, features=None, pydantic_model=None, is_conversation=False):
        if pydantic_model:
            return LLMService._call_structured(text, api_key, prompt_type, pydantic_model, is_conversation)
        return LLMService._call_text(text, api_key, prompt_type, is_conversation)
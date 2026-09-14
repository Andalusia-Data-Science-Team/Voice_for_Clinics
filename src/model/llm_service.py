import asyncio
import logging
from typing import Callable, List, Optional, Type
from openai import OpenAI
from pydantic import BaseModel, ValidationError, field_validator

from core.config import Config
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


def _mask(key: str) -> str:
    return f"...{key[-4:]}" if key and len(key) >= 4 else "(empty)"


def _is_key_failure(exc: Exception) -> bool:
    """
    True for errors that mean 'this specific key is bad/out of credit/rate
    limited' (HTTP 401/402/403/429) — i.e. worth retrying with a different
    key. False for anything else (bad request, model error, network issue,
    etc.) — those would fail identically on any key, so we don't burn through
    the whole pool retrying them.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    return status in (401, 402, 403, 429)


def _resolve_key_pool(primary_key: Optional[str]) -> List[str]:
    """
    Config.OPENROUTER_API_KEYS already has the primary key first (see
    core/config.py) followed by any configured fallbacks. Callers still pass
    their own `api_key` argument for backward compatibility / callers outside
    Config — if that happens to differ from the configured pool (or the pool
    is empty), make sure it's tried too, first.
    """
    pool = list(Config.OPENROUTER_API_KEYS)
    if primary_key and primary_key not in pool:
        pool.insert(0, primary_key)
    return pool


def _run_with_key_fallback(primary_key: Optional[str], make_call: Callable[[str], object]):
    """
    Synchronous fallback runner. make_call(api_key) -> result.
    Tries each key in the resolved pool in order; on a key-related failure
    (401/402/403/429) moves to the next key. Any other exception is raised
    immediately — it's not something a different key would fix.
    """
    pool = _resolve_key_pool(primary_key)
    if not pool:
        raise RuntimeError("No OpenRouter API key configured (OPENROUTER_API_KEY / OPENROUTER_API_KEYS)")

    last_exc: Optional[Exception] = None
    for i, key in enumerate(pool):
        try:
            return make_call(key)
        except Exception as exc:
            if _is_key_failure(exc):
                logger.warning(
                    "[LLMService] OpenRouter key %s (#%d/%d) failed: %s — trying next key",
                    _mask(key), i + 1, len(pool), exc,
                )
                last_exc = exc
                continue
            raise
    logger.error("[LLMService] All %d OpenRouter key(s) exhausted", len(pool))
    raise last_exc


async def _open_stream_with_key_fallback(primary_key: Optional[str], make_stream: Callable[[str], object]):
    """
    Async fallback runner for streaming calls. make_stream(api_key) opens the
    blocking streaming response object (run off-thread via asyncio.to_thread).

    Retrying only happens while OPENING the stream — before any chunk has been
    yielded to the caller. Once a key succeeds in opening the stream we commit
    to it for the rest of that stream; switching keys mid-stream would mean
    silently restarting output the client has already partially received.
    """
    pool = _resolve_key_pool(primary_key)
    if not pool:
        raise RuntimeError("No OpenRouter API key configured (OPENROUTER_API_KEY / OPENROUTER_API_KEYS)")

    last_exc: Optional[Exception] = None
    for i, key in enumerate(pool):
        try:
            return await asyncio.to_thread(make_stream, key)
        except Exception as exc:
            if _is_key_failure(exc):
                logger.warning(
                    "[LLMService] OpenRouter key %s (#%d/%d) failed opening stream: %s — trying next key",
                    _mask(key), i + 1, len(pool), exc,
                )
                last_exc = exc
                continue
            raise
    logger.error("[LLMService] All %d OpenRouter key(s) exhausted opening stream", len(pool))
    raise last_exc


# ---------------- LLM Service ---------------- #

class LLMService:

    DEEPSEEK_MODEL = "deepseek/deepseek-chat-v3-0324"   # fast DeepSeek for text tasks
    FAST_MODEL     = "openai/gpt-4o-mini"               # fast + reliable structured JSON

    # ── Non-streaming text methods ──────────────────────────────────────
    @staticmethod
    def refine_en_transcription(raw_text: str, api_key: str, is_conversation: bool = False, specialty: Optional[str] = None):
        return LLMService._call_text(raw_text, api_key, "refine_english", is_conversation, specialty=specialty)

    @staticmethod
    def refine_ar_transcription(raw_text: str, api_key: str, is_conversation: bool = False, specialty: Optional[str] = None):
        return LLMService._call_text(raw_text, api_key, "refine_arabic", is_conversation, specialty=specialty)

    @staticmethod
    def translate_to_eng(refined_text: str, api_key: str, is_conversation: bool = False):
        return LLMService._call_text(refined_text, api_key, "translate", is_conversation)

    # ── Streaming text methods ───────────────────────────────────────────
    @staticmethod
    async def refine_en_transcription_stream(raw_text: str, api_key: str, is_conversation: bool = False, specialty: Optional[str] = None):
        async for delta in LLMService._stream_text(raw_text, api_key, "refine_english", is_conversation, specialty=specialty):
            yield delta

    @staticmethod
    async def refine_ar_transcription_stream(raw_text: str, api_key: str, is_conversation: bool = False, specialty: Optional[str] = None):
        async for delta in LLMService._stream_text(raw_text, api_key, "refine_arabic", is_conversation, specialty=specialty):
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
    def _call_text(text: str, api_key: str, prompt_type: str, is_conversation: bool, specialty: Optional[str] = None) -> str:
        prompt = LLMService._get_prompt(prompt_type, text, is_conversation, specialty=specialty)

        def make_call(key: str) -> str:
            client = _get_client(key)
            response = client.chat.completions.create(
                model=LLMService.DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0,
            )
            return response.choices[0].message.content.strip() if response.choices else text

        return _run_with_key_fallback(api_key, make_call)

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

        def make_call(key: str):
            client = _get_client(key)
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
                # Not a key problem — a bad/malformed response from THIS key's
                # call would look the same on any other key too. Don't rotate
                # keys for this; surface it as a normal failure.
                logger.error(f"Structured output validation failed: {e}")
                logger.error(f"Raw output was: {raw_output[:500]}")
                return None

        return _run_with_key_fallback(api_key, make_call)

    # ── Internal: streaming text ──────────────────────────────────────────
    @staticmethod
    async def _stream_text(
        text: str,
        api_key: str,
        prompt_type: str,
        is_conversation: bool = False,
        specialty: Optional[str] = None,
    ):
        prompt = LLMService._get_prompt(prompt_type, text, is_conversation, specialty=specialty)

        def open_stream(key: str):
            client = _get_client(key)
            return client.chat.completions.create(
                model=LLMService.DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0,
                stream=True,
            )

        response = await _open_stream_with_key_fallback(api_key, open_stream)
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ── Prompt dispatch ───────────────────────────────────────────────────
    @staticmethod
    def _get_prompt(prompt_type: str, text: str, is_conversation: bool, specialty: Optional[str] = None) -> str:
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

        # Only the refinement prompts currently accept specialty context —
        # translate/extract/questions prompt functions take text only.
        if prompt_type in ("refine_english", "refine_arabic"):
            return func(text, specialty=specialty)
        return func(text)

    # ── Legacy process_text (kept for backward compat) ────────────────────
    @staticmethod
    def process_text(text, api_key, model, prompt_type, features=None, pydantic_model=None, is_conversation=False):
        if pydantic_model:
            return LLMService._call_structured(text, api_key, prompt_type, pydantic_model, is_conversation)
        return LLMService._call_text(text, api_key, prompt_type, is_conversation)
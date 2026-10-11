import logging
import os
import time
from typing import Any, Optional, Type
from pydantic import BaseModel, ValidationError

from core.model_registry import (
    DIAGNOSIS_MAX_TOKENS,
    EXTRACTION_MAX_TOKENS,
    FOUNDRY_DEFAULT_EXTRACTION_MODEL,
    REASONING_EFFORT,
    ModelSpec,
    get_model,
)
from llm.provider import build_foundry_kwargs
from utils import prompt as prompt_utils
from utils import clinical_prompts
from utils.utils import safe_parse_json
from llm.runtime import LLMRuntime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------- Helpers ------------------------------------------------- #

def _active_provider() -> str:
    return os.environ.get("LLM_PROVIDER", "openrouter").lower()


# ---------------- Pydantic Models ---------------- #

class ExtractedFeatures(BaseModel):
    """Shared wrapper for both structured steps: {"json_data": {...}, "summary": "..."}"""
    json_data: dict


def _record_meta(meta: Optional[dict], response, model: ModelSpec, latency_sec: float, used_reasoning: bool) -> None:
    """Fills `meta` (if the caller passed one) with what the evaluation suite and
    usage log need: model, latency of the LLM call, token counts, and OpenRouter's
    own cost figure (`usage.cost`, requested via usage.include) when present."""
    if meta is None:
        return
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    meta.update({
        "model_id": model.id,
        "model_slug": model.slug,
        "latency_sec": round(latency_sec, 3),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None),
        "cost_usd": getattr(usage, "cost", None),
        "reasoning_enabled": used_reasoning,
    })


# ---------------- LLM Service ---------------- #

class LLMService:

    DEEPSEEK_MODEL = "deepseek/deepseek-chat-v3-0324"   # fast DeepSeek for text tasks (refine / translate)

    # ── Non-streaming text methods ──────────────────────────────────────
    @staticmethod
    def refine_en_transcription(raw_text: str, api_key: str, is_conversation: bool = False, specialty: Optional[str] = None):
        """Deprecated synchronous entry point; use the async runtime method."""
        raise RuntimeError("Use refine_en_transcription_async with the shared LLMRuntime")

    @staticmethod
    def refine_ar_transcription(raw_text: str, api_key: str, is_conversation: bool = False, specialty: Optional[str] = None):
        """Deprecated synchronous entry point; use the async runtime method."""
        raise RuntimeError("Use refine_ar_transcription_async with the shared LLMRuntime")

    @staticmethod
    def translate_to_eng(refined_text: str, api_key: str, is_conversation: bool = False):
        """Deprecated synchronous entry point; use the async runtime method."""
        raise RuntimeError("Use translate_to_eng_async with the shared LLMRuntime")

    # ── Streaming text methods ───────────────────────────────────────────
    @staticmethod
    async def refine_en_transcription_stream(raw_text: str, api_key: str, is_conversation: bool = False, specialty: Optional[str] = None):
        """Deprecated stream entry point; use the async runtime method."""
        raise RuntimeError("Use refine_en_transcription_stream_async with the shared LLMRuntime")

    @staticmethod
    async def refine_ar_transcription_stream(raw_text: str, api_key: str, is_conversation: bool = False, specialty: Optional[str] = None):
        """Deprecated stream entry point; use the async runtime method."""
        raise RuntimeError("Use refine_ar_transcription_stream_async with the shared LLMRuntime")

    @staticmethod
    async def translate_to_eng_stream(refined_text: str, api_key: str, is_conversation: bool = False):
        """Deprecated stream entry point; use the async runtime method."""
        raise RuntimeError("Use translate_to_eng_stream_async with the shared LLMRuntime")

    @staticmethod
    async def refine_en_transcription_async(raw_text: str, runtime: LLMRuntime, is_conversation: bool = False, specialty: Optional[str] = None) -> str:
        """Refine English text through the shared asynchronous runtime."""
        return await LLMService._call_text_async(raw_text, runtime, "refine_english", is_conversation, specialty)

    @staticmethod
    async def refine_ar_transcription_async(raw_text: str, runtime: LLMRuntime, is_conversation: bool = False, specialty: Optional[str] = None) -> str:
        """Refine Arabic text through the shared asynchronous runtime."""
        return await LLMService._call_text_async(raw_text, runtime, "refine_arabic", is_conversation, specialty)

    @staticmethod
    async def translate_to_eng_async(refined_text: str, runtime: LLMRuntime, is_conversation: bool = False) -> str:
        """Translate text through the shared asynchronous runtime."""
        return await LLMService._call_text_async(refined_text, runtime, "translate", is_conversation)

    @staticmethod
    async def refine_en_transcription_stream_async(raw_text: str, runtime: LLMRuntime, is_conversation: bool = False, specialty: Optional[str] = None):
        """Stream refined English text without blocking the event loop."""
        async for delta in LLMService._stream_text_async(raw_text, runtime, "refine_english", is_conversation, specialty):
            yield delta

    @staticmethod
    async def refine_ar_transcription_stream_async(raw_text: str, runtime: LLMRuntime, is_conversation: bool = False, specialty: Optional[str] = None):
        """Stream refined Arabic text without blocking the event loop."""
        async for delta in LLMService._stream_text_async(raw_text, runtime, "refine_arabic", is_conversation, specialty):
            yield delta

    @staticmethod
    async def translate_to_eng_stream_async(refined_text: str, runtime: LLMRuntime, is_conversation: bool = False):
        """Stream English translation without blocking the event loop."""
        async for delta in LLMService._stream_text_async(refined_text, runtime, "translate", is_conversation):
            yield delta

    # ── Structured step 1: clinical note field extraction ────────────────
    @staticmethod
    def extract_features(
        translated_text: str,
        api_key: str,
        department: str = "opd",
        is_conversation: bool = False,
        model_id: Optional[str] = None,
        meta: Optional[dict] = None,
    ):
        """Field extraction only (no diagnoses / ICD-10 / history). `model_id` is
        one of core.model_registry.EXTRACTION_MODELS (None -> default). Pass a
        dict as `meta` to receive latency / token / cost info for the call."""
        model = get_model("extraction", model_id)
        prompt = clinical_prompts.get_extraction_prompt(department, translated_text, is_conversation)
        return LLMService._call_structured(
            prompt, ExtractedFeatures, api_key, model,
            max_tokens=EXTRACTION_MAX_TOKENS, reasoning_effort=None, meta=meta,
        )

    # ── Structured step 2: ICD-10 recommendation + reasoning ─────────────
    @staticmethod
    def recommend_diagnosis(
        translated_text: str,
        api_key: str,
        department: str = "opd",
        patient_history_text: str = "",
        is_conversation: bool = False,
        model_id: Optional[str] = None,
        meta: Optional[dict] = None,
    ):
        """ICD-10 codes + reasoning (ER: + differential diagnoses + triage).
        `model_id` is one of core.model_registry.REASONING_MODELS. Reasoning-
        capable models run with OpenRouter's `reasoning` param enabled."""
        model = get_model("reasoning", model_id)
        prompt = clinical_prompts.get_diagnosis_prompt(department, translated_text, is_conversation, patient_history_text)
        return LLMService._call_structured(
            prompt, ExtractedFeatures, api_key, model,
            max_tokens=DIAGNOSIS_MAX_TOKENS, reasoning_effort=REASONING_EFFORT, meta=meta,
        )

    @staticmethod
    async def extract_features_async(translated_text: str, runtime: LLMRuntime, department: str = "opd", is_conversation: bool = False, model_id: Optional[str] = None, meta: Optional[dict] = None):
        """Extract clinical fields through the shared asynchronous runtime."""
        model = get_model("extraction", model_id)
        prompt = clinical_prompts.get_extraction_prompt(department, translated_text, is_conversation)
        return await LLMService._call_structured_async(prompt, ExtractedFeatures, runtime, model, EXTRACTION_MAX_TOKENS, meta=meta)

    @staticmethod
    async def recommend_diagnosis_async(translated_text: str, runtime: LLMRuntime, department: str = "opd", patient_history_text: str = "", is_conversation: bool = False, model_id: Optional[str] = None, meta: Optional[dict] = None):
        """Recommend diagnoses through the shared asynchronous runtime."""
        model = get_model("reasoning", model_id)
        prompt = clinical_prompts.get_diagnosis_prompt(department, translated_text, is_conversation, patient_history_text)
        return await LLMService._call_structured_async(prompt, ExtractedFeatures, runtime, model, DIAGNOSIS_MAX_TOKENS, reasoning_effort=REASONING_EFFORT, meta=meta)

    # ── Internal: plain text call (refine / translate) ───────────────────
    @staticmethod
    def _call_text(text: str, api_key: str, prompt_type: str, is_conversation: bool, specialty: Optional[str] = None) -> str:
        """Reject removed synchronous calls instead of constructing a client."""
        raise RuntimeError("Synchronous LLM calls were removed; use the async runtime")

    @staticmethod
    async def _call_text_async(text: str, runtime: LLMRuntime, prompt_type: str, is_conversation: bool, specialty: Optional[str] = None) -> str:
        prompt = LLMService._get_prompt(prompt_type, text, is_conversation, specialty=specialty)

        if _active_provider() == "foundry":
            # Refine/translate use the Luna deployment on Foundry
            # (DeepSeek is OpenRouter-only)
            deployment = FOUNDRY_DEFAULT_EXTRACTION_MODEL
            kwargs = build_foundry_kwargs(
                deployment,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                stream=False,
            )
            response = await runtime.complete(model=deployment, create_kwargs=kwargs)
        else:
            response = await runtime.complete(
                model=LLMService.DEEPSEEK_MODEL,
                create_kwargs={"model": LLMService.DEEPSEEK_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048, "temperature": 0},
            )
        return response.choices[0].message.content.strip() if response.choices and response.choices[0].message.content else text

    # ── Internal: structured JSON call ───────────────────────────────────
    @staticmethod
    def _call_structured(
        prompt: str,
        pydantic_model: Type[BaseModel],
        api_key: str,
        model: ModelSpec,
        max_tokens: int,
        reasoning_effort: Optional[str] = None,
        meta: Optional[dict] = None,
    ):
        """Reject removed synchronous calls instead of constructing a client."""
        raise RuntimeError("Synchronous LLM calls were removed; use the async runtime")

    @staticmethod
    async def _call_structured_async(prompt: str, pydantic_model: Type[BaseModel], runtime: LLMRuntime, model: ModelSpec, max_tokens: int, reasoning_effort: Optional[str] = None, meta: Optional[dict] = None):
        use_reasoning = bool(reasoning_effort and model.supports_reasoning)
        t0 = time.perf_counter()

        if _active_provider() == "foundry":
            # ── Foundry path ─────────────────────────────────────────────────
            # No extra_body (usage.include / reasoning are OpenRouter-specific).
            # Use json_schema_strict where supported, fall back to json_object.
            deployment = model.foundry_deployment
            if not deployment:
                raise RuntimeError(
                    f"Model {model.id!r} has no Foundry deployment configured. "
                    "Select a model with a Foundry deployment or use LLM_PROVIDER=openrouter."
                )
            kwargs = build_foundry_kwargs(
                deployment,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                stream=False,
                response_format={"type": "json_object"},
                reasoning_effort=reasoning_effort if use_reasoning else None,
            )
            model_key = deployment  # used for limiter, retry, telemetry
        else:
            # ── OpenRouter path (unchanged) ───────────────────────────────────
            extra_body: dict[str, Any] = {"usage": {"include": True}}
            if use_reasoning:
                extra_body["reasoning"] = {"effort": reasoning_effort}
            kwargs = {
                "model": model.slug,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "extra_body": extra_body,
                "response_format": {"type": "json_object"},
            }
            if not use_reasoning:
                kwargs["temperature"] = 0
            model_key = model.slug

        try:
            response = await runtime.complete(model=model_key, create_kwargs=kwargs)
        except Exception as exc:
            if getattr(exc, "status_code", None) != 400:
                raise
            # 400 means the model rejected response_format — retry without it
            kwargs.pop("response_format", None)
            response = await runtime.complete(model=model_key, create_kwargs=kwargs)

        _record_meta(meta, response, model, time.perf_counter() - t0, use_reasoning)
        choice = response.choices[0] if response.choices else None
        raw_output = ((choice.message.content if choice else "") or "").strip()
        if not raw_output:
            return None
        try:
            parsed = safe_parse_json(raw_output)
            if isinstance(parsed, dict) and len(parsed) == 1:
                inner = parsed[next(iter(parsed))]
                if isinstance(inner, dict) and set(pydantic_model.model_fields) & set(inner):
                    parsed = inner
            return pydantic_model(**parsed).model_dump()
        except (ValueError, ValidationError) as exc:
            if meta is not None:
                meta["error"] = f"unparseable output: {exc}"
            return None

    # ── Internal: streaming text ──────────────────────────────────────────
    @staticmethod
    async def _stream_text(
        text: str,
        api_key: str,
        prompt_type: str,
        is_conversation: bool = False,
        specialty: Optional[str] = None,
    ):
        """Reject removed synchronous streams instead of blocking the event loop."""
        raise RuntimeError("Synchronous LLM streams were removed; use the async runtime")

    @staticmethod
    async def _stream_text_async(text: str, runtime: LLMRuntime, prompt_type: str, is_conversation: bool = False, specialty: Optional[str] = None):
        prompt = LLMService._get_prompt(prompt_type, text, is_conversation, specialty=specialty)

        if _active_provider() == "foundry":
            deployment = FOUNDRY_DEFAULT_EXTRACTION_MODEL
            kwargs = build_foundry_kwargs(
                deployment,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                stream=True,
            )
            async for delta in runtime.stream(model=deployment, create_kwargs=kwargs):
                yield delta
        else:
            async for delta in runtime.stream(
                model=LLMService.DEEPSEEK_MODEL,
                create_kwargs={"model": LLMService.DEEPSEEK_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048, "temperature": 0},
            ):
                yield delta

    # ── Prompt dispatch (refine / translate only) ─────────────────────────
    @staticmethod
    def _get_prompt(prompt_type: str, text: str, is_conversation: bool, specialty: Optional[str] = None) -> str:
        mapping = {
            ("refine_english", False): prompt_utils.get_refine_english_prompt_deepseek,
            ("refine_english", True):  prompt_utils.get_refine_english_prompt_deepseek_conversation,
            ("refine_arabic",  False): prompt_utils.get_refine_arabic_prompt_deepseek,
            ("refine_arabic",  True):  prompt_utils.get_refine_arabic_prompt_deepseek_conversation,
            ("translate",      False): prompt_utils.get_translation_prompt_deepseek,
            ("translate",      True):  prompt_utils.get_translation_prompt_deepseek_conversation,
        }
        func = mapping.get((prompt_type, is_conversation))
        if func is None:
            raise ValueError(f"Unsupported prompt_type={prompt_type}, is_conversation={is_conversation}")
        # Only the refinement prompts accept specialty context.
        if prompt_type in ("refine_english", "refine_arabic"):
            return func(text, specialty=specialty)
        return func(text)
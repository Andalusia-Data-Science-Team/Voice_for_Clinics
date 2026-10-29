"""
tests/phase1/test_m2.py — Track M2: Provider Adapter Verification Suite
Covers:
  - M2.1: Provider interfaces (5 hard tests)
  - M2.2: Config switch OpenRouter -> Foundry (5 hard tests)
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from providers.base import LLMProvider, ProviderHealthResult, STTProvider
from providers.factory import (
    build_llm_provider,
    build_stt_provider,
    get_llm,
    get_stt,
    reset_providers,
)
from providers.foundry_llm import FoundryLLMProvider
from providers.munsit_stt import MunsitSTTProvider
from providers.openrouter_llm import OpenRouterLLMProvider
from providers.voxtral_stt import VoxtralSTTProvider
from tests.phase1.harness import SubtaskRunner


# ==============================================================================
# M2.1 — Provider interfaces
# ==============================================================================
def run_test_m2_1() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="M2",
        subtask_id="M2.1",
        title="Provider interfaces (LLM and STT abstract base classes & contracts)",
        target_files=["src/providers/base.py", "src/providers/factory.py"],
        source_file=__file__,
    )

    # TC 1: ABC enforcement — Instantiate BaseLLMProvider -> TypeError raised
    def tc1():
        with pytest.raises(TypeError) as exc_info:
            LLMProvider()  # type: ignore
        assert "abstract class" in str(exc_info.value).lower()
        return True, "Instantiating LLMProvider ABC raises TypeError as expected"

    runner.record_case(
        case_num=1,
        name="ABC instantiation enforcement",
        input_setup="Attempt direct instantiation of abstract class LLMProvider()",
        expected_result="TypeError raised: Can't instantiate abstract class with abstract methods",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: Abstract methods — Inspect __abstractmethods__ -> complete, stream, health_check
    def tc2():
        methods = LLMProvider.__abstractmethods__
        required = {"complete", "stream", "health_check", "close"}
        assert required.issubset(methods), f"Missing required abstract methods: {required - methods}"
        return True, f"All required abstract methods enforced: {sorted(list(methods))}"

    runner.record_case(
        case_num=2,
        name="Abstract methods contract inspection",
        input_setup="Inspect LLMProvider.__abstractmethods__",
        expected_result="complete, stream, health_check, and close are declared abstract",
        severity="Crit",
        fn=tc2,
    )

    # TC 3: HealthResult immutable — Modify ok field -> FrozenInstanceError
    def tc3():
        res = ProviderHealthResult.healthy(provider="foundry", latency_ms=12.5)
        with pytest.raises(Exception):
            res.ok = False  # type: ignore
        return True, "ProviderHealthResult is an immutable frozen dataclass; mutations rejected"

    runner.record_case(
        case_num=3,
        name="HealthResult immutability",
        input_setup="Attempt to mutate 'ok' attribute on ProviderHealthResult instance",
        expected_result="FrozenInstanceError / exception raised on mutation attempt",
        severity="High",
        fn=tc3,
    )

    # TC 4: Foundry implements all — Instantiate FoundryLLMProvider -> No TypeError
    def tc4():
        mock_runtime = MagicMock()
        provider = FoundryLLMProvider(runtime=mock_runtime)
        assert isinstance(provider, LLMProvider)
        assert provider.provider_name == "foundry"
        return True, "FoundryLLMProvider successfully implements all abstract methods without TypeError"

    runner.record_case(
        case_num=4,
        name="Foundry provider contract implementation",
        input_setup="Instantiate concrete FoundryLLMProvider",
        expected_result="Instantiates cleanly without TypeError; satisfies all LLMProvider contracts",
        severity="Crit",
        fn=tc4,
    )

    # TC 5: STT contracts — Munsit + Voxtral -> Both have transcribe_live, health_check
    def tc5():
        munsit_methods = dir(MunsitSTTProvider)
        voxtral_methods = dir(VoxtralSTTProvider)
        for req in ["transcribe_live_ar", "transcribe_live_en", "transcribe_file", "health_check"]:
            assert req in munsit_methods, f"Munsit missing {req}"
            assert req in voxtral_methods, f"Voxtral missing {req}"
        return True, "Both MunsitSTTProvider and VoxtralSTTProvider implement all STT interface methods"

    runner.record_case(
        case_num=5,
        name="STT provider contracts validation",
        input_setup="Inspect MunsitSTTProvider and VoxtralSTTProvider method interfaces",
        expected_result="Both providers implement transcribe_live_ar, transcribe_live_en, and health_check",
        severity="Crit",
        fn=tc5,
    )

    return runner


# ==============================================================================
# M2.2 — Config switch OpenRouter -> Foundry
# ==============================================================================
def run_test_m2_2() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="M2",
        subtask_id="M2.2",
        title="Configuration switch OpenRouter -> Foundry (zero-code environment switching)",
        target_files=["src/providers/factory.py"],
        source_file=__file__,
    )

    # TC 1: Default = Foundry — No LLM_PROVIDER env -> FoundryLLMProvider
    def tc1():
        reset_providers()
        with patch.dict(os.environ, {}, clear=True):
            # No LLM_PROVIDER set; factory default is foundry
            with patch("llm.runtime.LLMRuntime"):
                provider = build_llm_provider()
                assert isinstance(provider, FoundryLLMProvider)
                assert provider.provider_name == "foundry"
                return True, "Default provider without env override is FoundryLLMProvider"

    runner.record_case(
        case_num=1,
        name="Default provider resolves to Foundry",
        input_setup="Call build_llm_provider() without LLM_PROVIDER environment variable set",
        expected_result="Returns FoundryLLMProvider as the default enterprise provider",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: Switch to OpenRouter — LLM_PROVIDER=openrouter -> OpenRouterLLMProvider
    def tc2():
        reset_providers()
        with patch.dict(os.environ, {"LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "test-key-123"}):
            provider = build_llm_provider()
            assert isinstance(provider, OpenRouterLLMProvider)
            assert provider.provider_name == "openrouter"
            return True, "Switched cleanly to OpenRouterLLMProvider via LLM_PROVIDER=openrouter"

    runner.record_case(
        case_num=2,
        name="Environment switch to OpenRouter",
        input_setup="Set LLM_PROVIDER=openrouter in environment and invoke build_llm_provider()",
        expected_result="Returns OpenRouterLLMProvider instance without code modifications",
        severity="High",
        fn=tc2,
    )

    # TC 3: Invalid provider — LLM_PROVIDER=nonexistent -> ValueError with valid list
    def tc3():
        reset_providers()
        with patch.dict(os.environ, {"LLM_PROVIDER": "nonexistent_vendor"}):
            with pytest.raises(ValueError) as exc:
                build_llm_provider()
            assert "unknown llm_provider" in str(exc.value).lower()
            assert "foundry" in str(exc.value).lower()
            return True, f"Invalid provider rejected with descriptive error: {exc.value}"

    runner.record_case(
        case_num=3,
        name="Invalid provider rejection with valid choices",
        input_setup="Set LLM_PROVIDER=nonexistent_vendor and invoke build_llm_provider()",
        expected_result="Raises ValueError detailing valid supported providers",
        severity="High",
        fn=tc3,
    )

    # TC 4: STT switch — STT_PROVIDER=voxtral -> VoxtralSTTProvider
    def tc4():
        reset_providers()
        with patch.dict(os.environ, {"STT_PROVIDER": "voxtral", "MISTRAL_API_KEY": "test-key"}):
            stt = build_stt_provider()
            assert isinstance(stt, VoxtralSTTProvider)
            assert stt.provider_name == "voxtral"
            return True, "STT provider switched seamlessly to VoxtralSTTProvider via environment"

    runner.record_case(
        case_num=4,
        name="STT provider switch to Voxtral",
        input_setup="Set STT_PROVIDER=voxtral and invoke build_stt_provider()",
        expected_result="Returns VoxtralSTTProvider cleanly",
        severity="High",
        fn=tc4,
    )

    # TC 5: Runtime switch — Change env; re-call factory -> New provider; old not cached
    def tc5():
        reset_providers()
        # Step 1: OpenRouter
        with patch.dict(os.environ, {"LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "key1"}):
            p1 = get_llm()
            assert isinstance(p1, OpenRouterLLMProvider)

        # Step 2: Switch to Foundry and reset
        reset_providers()
        with patch.dict(os.environ, {"LLM_PROVIDER": "foundry"}):
            with patch("llm.runtime.LLMRuntime"):
                p2 = get_llm()
                assert isinstance(p2, FoundryLLMProvider)
                assert id(p1) != id(p2)
                return True, "Runtime reset and re-invocation correctly returns new provider without stale cache"

    runner.record_case(
        case_num=5,
        name="Dynamic runtime switch without stale cache",
        input_setup="Instantiate provider, reset singleton cache, update env, and re-query factory",
        expected_result="New provider instantiated immediately; old cached instance discarded",
        severity="Crit",
        fn=tc5,
    )

    return runner

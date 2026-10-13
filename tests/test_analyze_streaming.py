import asyncio

import pytest
from starlette.requests import Request

import app as app_module


def _event_name(event: dict) -> str:
    return event["event"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fast_step", ["extraction", "diagnosis"])
async def test_each_parallel_result_is_sent_without_waiting_for_the_other(monkeypatch, fast_step):
    slow_step_started = asyncio.Event()
    release_slow_step = asyncio.Event()

    async def extract_features_async(**kwargs):
        if fast_step == "diagnosis":
            slow_step_started.set()
            await release_slow_step.wait()
        return {"json_data": {"chief_complaint": "Chest pain"}}

    async def recommend_diagnosis_async(**kwargs):
        if fast_step == "extraction":
            slow_step_started.set()
            await release_slow_step.wait()
        return {"json_data": {"icd10_codes": ["R07.9"]}}

    monkeypatch.setattr(app_module.LLMService, "extract_features_async", extract_features_async)
    monkeypatch.setattr(app_module.LLMService, "recommend_diagnosis_async", recommend_diagnosis_async)
    monkeypatch.setattr(
        app_module.QuestionGenerator,
        "generate",
        lambda **kwargs: ([], None),
    )
    monkeypatch.setattr(app_module, "save_usage_log", lambda entry: None)

    request = Request({
        "type": "http",
        "method": "POST",
        "scheme": "http",
        "path": "/audio/analyze",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 9999),
    })
    response = await app_module.analyze_transcript(
        request=request,
        runtime=object(),
        text="The patient reports chest pain.",
        language=app_module.Language.english,
        mode=app_module.Mode.doctor,
        department=app_module.Department.opd,
        specialty=None,
        use_history=False,
        patient_id=None,
        business_unit=None,
        clinician_name=None,
        transcription_sec=None,
        session_id="test-session",
    )

    events = response.body_iterator.__aiter__()
    assert _event_name(await anext(events)) == "refinement_skipped"
    assert _event_name(await asyncio.wait_for(anext(events), timeout=1)) == fast_step
    await asyncio.wait_for(slow_step_started.wait(), timeout=1)
    assert not release_slow_step.is_set()

    release_slow_step.set()
    slow_step = "diagnosis" if fast_step == "extraction" else "extraction"
    assert _event_name(await asyncio.wait_for(anext(events), timeout=1)) == slow_step

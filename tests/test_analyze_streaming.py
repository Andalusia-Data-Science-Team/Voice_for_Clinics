import asyncio
import json

import pytest
from starlette.requests import Request

import app as app_module


async def anext(iterator):
    return await iterator.__anext__()


def _event_name(event: dict) -> str:
    return event["event"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fast_step", ["extraction", "diagnosis"])
async def test_each_parallel_result_is_sent_without_waiting_for_the_other(monkeypatch, fast_step):
    slow_step_started = asyncio.Event()
    release_slow_step = asyncio.Event()
    selected_candidates = []

    async def extract_features_async(**kwargs):
        if fast_step == "diagnosis":
            slow_step_started.set()
            await release_slow_step.wait()
        return {"json_data": {"chief_complaint": "Chest pain"}}

    async def recommend_diagnosis_async(**kwargs):
        if fast_step == "extraction":
            slow_step_started.set()
            await release_slow_step.wait()
        selected_candidates.extend(kwargs["catalogue_candidates"][:1])
        code = selected_candidates[0][0]
        return {
            "json_data": {
                "icd10_codes": [
                    f"{code} - Model-invented diagnosis text",
                    "FAKE999.00 - Unsupported code",
                ],
                "final_diagnosis": "Unsupported diagnosis text",
            }
        }

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
        inp=app_module.AnalyzeInput(
            text="The patient reports chest pain.",
            language="en",
            mode="doctor",
            department="opd",
            session_id="test-session",
        ),
    )

    events = response.body_iterator.__aiter__()
    assert _event_name(await anext(events)) == "refinement_skipped"
    routing_event = await asyncio.wait_for(anext(events), timeout=1)
    assert _event_name(routing_event) == "routing_signal"
    routing_data = json.loads(routing_event["data"])
    assert routing_data["requires_clinician_review"] is True
    fast_event = await asyncio.wait_for(anext(events), timeout=1)
    assert _event_name(fast_event) == fast_step
    if fast_step == "extraction":
        fast_data = json.loads(fast_event["data"])
        assert fast_data["routing_signal"]["requires_clinician_review"] is True
    await asyncio.wait_for(slow_step_started.wait(), timeout=1)
    assert not release_slow_step.is_set()

    release_slow_step.set()
    slow_step = "diagnosis" if fast_step == "extraction" else "extraction"
    slow_event = await asyncio.wait_for(anext(events), timeout=1)
    assert _event_name(slow_event) == slow_step
    if slow_step == "extraction":
        slow_data = json.loads(slow_event["data"])
        assert slow_data["routing_signal"]["requires_clinician_review"] is True
    diagnosis_event = fast_event if fast_step == "diagnosis" else slow_event
    diagnosis_data = json.loads(diagnosis_event["data"])["diagnosis"]
    expected = f"{selected_candidates[0][0]} - {selected_candidates[0][1]}"
    assert diagnosis_data["icd10_codes"] == [expected]
    assert diagnosis_data["final_diagnosis"] == expected
    assert diagnosis_data["icd10_unmatched"] == [
        "FAKE999.00 - Unsupported code"
    ]

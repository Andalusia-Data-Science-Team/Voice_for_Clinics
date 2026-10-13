import pytest

import app as app_module


@pytest.mark.asyncio
async def test_refine_field_accepts_specialty_enum(monkeypatch):
    received = {}

    async def refine_en(text, runtime, *, is_conversation, specialty):
        received["specialty"] = specialty
        return text

    monkeypatch.setattr(app_module.LLMService, "refine_en_transcription_async", refine_en)

    result = await app_module.refine_field_answer(
        runtime=object(),
        text="The patient has hypertension.",
        language=app_module.Language.english,
        specialty=app_module.Specialty.Cardiology,
    )

    assert received["specialty"] == "Cardiology"
    assert result == {"refined_text": "The patient has hypertension."}

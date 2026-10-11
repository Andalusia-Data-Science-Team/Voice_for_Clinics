from core.model_registry import (
    DEFAULT_EXTRACTION_MODEL,
    DEFAULT_REASONING_MODEL,
    get_model,
    registry_for_ui,
)


def test_extraction_and_reasoning_use_gpt_56_luna():
    extraction = get_model("extraction", None)
    reasoning = get_model("reasoning", None)

    assert DEFAULT_EXTRACTION_MODEL == "gpt-5.6-luna"
    assert DEFAULT_REASONING_MODEL == "gpt-5.6-luna"
    assert extraction.id == reasoning.id == "gpt-5.6-luna"
    assert extraction.slug == reasoning.slug == "openai/gpt-5.6-luna"


def test_ui_registry_only_exposes_the_fixed_model():
    registry = registry_for_ui()

    for step in ("extraction", "reasoning"):
        assert registry[step]["default"] == "gpt-5.6-luna"
        assert registry[step]["models"] == [
            {"id": "gpt-5.6-luna", "label": "GPT 5.6 Luna"}
        ]

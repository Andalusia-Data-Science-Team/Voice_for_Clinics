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
    assert extraction.foundry_deployment == reasoning.foundry_deployment == "gpt-5.6-luna"


def test_ui_registry_only_exposes_the_fixed_model():
    registry = registry_for_ui()

    for step in ("extraction", "reasoning"):
        assert registry[step]["default"] == "gpt-5.6-luna"
        model_ids = [m["id"] for m in registry[step]["models"]]
        assert "gpt-5.6-luna" in model_ids
        assert "DeepSeek-V3.2" in model_ids

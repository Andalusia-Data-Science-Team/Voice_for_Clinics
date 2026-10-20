"""Public Pipeline Package Interface.

Provides:
- run_pipeline: Asynchronously executes the clinical pipeline and streams SSE events
- PipelineOrchestrator: The execution engine
- PipelineContext: The stateless context
- MergedResult: The combined execution outcome
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Callable, Coroutine, Dict, Optional

from llm.runtime import LLMRuntime
from core.model_registry import ModelSpec
from schemas.input_schemas import Language, Mode
from pipeline.merger import DiagnosisStepResult, ExtractionStepResult, MergedResult
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.steps import (
    PipelineContext,
    step_diagnose,
    step_extract,
    step_red_flags,
    step_refine_stream,
    step_translate_stream,
)

__all__ = [
    "run_pipeline",
    "PipelineOrchestrator",
    "PipelineContext",
    "MergedResult",
    "ExtractionStepResult",
    "DiagnosisStepResult",
    "step_refine_stream",
    "step_translate_stream",
    "step_red_flags",
    "step_extract",
    "step_diagnose",
]


def _format_sse_event(event_dict: Dict[str, Any]) -> Dict[str, str]:
    """Ensure event data payload is serialized properly for EventSourceResponse."""
    ev = event_dict.get("event", "message")
    data = event_dict.get("data", "")
    if isinstance(data, (dict, list)):
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    else:
        payload = str(data)
    return {"event": ev, "data": payload}


async def run_pipeline(
    ctx: PipelineContext,
    on_complete: Optional[Callable[[MergedResult], Coroutine[Any, Any, None]]] = None,
) -> AsyncGenerator[Dict[str, str], None]:
    """Run post-transcription pipeline and yield formatted SSE events."""
    orchestrator = PipelineOrchestrator(on_complete=on_complete)
    async for event in orchestrator.run(ctx):
        yield _format_sse_event(event)

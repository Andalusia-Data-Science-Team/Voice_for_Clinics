# MedVoice Engineering Documentation Hub

This directory contains deep architectural, technical, and operational documentation for the engineering and data science teams working on **MedVoice**.

---

## Documentation Index

### 1. Provider Abstraction Layer & Foundry Migration (M2)
- **[`docs/m2-provider-abstraction/01_architecture_and_contracts.md`](file:///home/ai/Workspace/Doha/MedVoice/docs/m2-provider-abstraction/01_architecture_and_contracts.md)**: Abstract base classes (`LLMProvider`, `STTProvider`), `ProviderHealthResult` value object, runtime validation, and design patterns.
- **[`docs/m2-provider-abstraction/02_concrete_adapters.md`](file:///home/ai/Workspace/Doha/MedVoice/docs/m2-provider-abstraction/02_concrete_adapters.md)**: In-depth details on `FoundryLLMProvider`, `MunsitSTTProvider`, and `VoxtralSTTProvider` delegation, argument transformation, and probe mechanics.
- **[`docs/m2-provider-abstraction/03_factory_and_configuration.md`](file:///home/ai/Workspace/Doha/MedVoice/docs/m2-provider-abstraction/03_factory_and_configuration.md)**: Zero-code provider switching via `LLM_PROVIDER` and `STT_PROVIDER`, FastAPI lifespan hooks, and backward-compatible singleton access.
- **[`docs/m2-provider-abstraction/04_testing_ci_and_observability.md`](file:///home/ai/Workspace/Doha/MedVoice/docs/m2-provider-abstraction/04_testing_ci_and_observability.md)**: CI gate test suite (`test_m2_*.py`), mocking strategies, `/health` endpoint integration, and latency tracking.

---

### 2. Schema Validation, MongoDB Persistence & Gold Evaluation Engine
- **[`docs/persistence-and-eval/01_schema_validation_architecture.md`](file:///home/ai/Workspace/Doha/MedVoice/docs/persistence-and-eval/01_schema_validation_architecture.md)**: Comprehensive Pydantic v2 schemas (`input_schemas.py`, `output_schemas.py`), department validation (ER, OPD, Dental), and sanitized audit trails.
- **[`docs/persistence-and-eval/02_mongodb_session_persistence.md`](file:///home/ai/Workspace/Doha/MedVoice/docs/persistence-and-eval/02_mongodb_session_persistence.md)**: Data models (`SessionDocument`, `PipelineRunDocument`, `FeedbackDocument`), async motor connection, write-behind buffering, and sub-millisecond compound indexing.
- **[`docs/persistence-and-eval/03_gold_dataset_and_eval_engine.md`](file:///home/ai/Workspace/Doha/MedVoice/docs/persistence-and-eval/03_gold_dataset_and_eval_engine.md)**: Ground-truth clinical gold case schemas, automated evaluation runner, clinical metrics (BLEU, ROUGE-L, Levenshtein, field accuracy), and CLI harness (`run_eval.py`).
- **[`docs/persistence-and-eval/04_performance_and_hardening.md`](file:///home/ai/Workspace/Doha/MedVoice/docs/persistence-and-eval/04_performance_and_hardening.md)**: Connection pooling calibration, write batching throughput, schema compilation caching, and graceful fallback to JSONL.

---

### Additional Core References
- [`docs/PROJECT_DOCUMENTATION.md`](file:///home/ai/Workspace/Doha/MedVoice/docs/PROJECT_DOCUMENTATION.md): Overview of clinical speech pipeline and deployment topologies.
- [`docs/llm_concurrency_README.md`](file:///home/ai/Workspace/Doha/MedVoice/docs/llm_concurrency_README.md): Concurrency limiters, backoff jitter, and per-model token buckets.

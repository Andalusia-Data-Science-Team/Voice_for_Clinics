# CI/CD and staging (A4)

`azure-pipelines.yml` runs on `main` and pull requests to `main`. CI installs `src/requirements.txt` plus `requirements-test.txt`, checks Ruff runtime-error rules, then runs the concurrency and full suites in a clean environment (`env -i`, `PYTHON_DOTENV_DISABLED=1`) with inline dummy Foundry configuration for three provider configurations: `LLM_PROVIDER` unset (default), `openrouter`, and `foundry`. It never reads or publishes `.env`. `tests/test_m1_verification.py` is explicitly ignored because it loads `src/.env` and calls live Azure infrastructure. A successful main build uses the existing `Dockerfile` and pushes Build ID and short SHA tags.

## Deployment uses the same Docker Compose stack as local development

All dev, staging, and production deployments use `pipelines/deploy-template.yml`, which deploys the **same `docker-compose.yml`** developers run locally. Only the environment name, variable group, image tag, and compose directory differ.

The deploy step:

1. Verifies the pre-existing server env file is readable and the compose checkout exists. It never creates, copies, or prints the env file.
2. Records the image the stack is currently running (`docker inspect` on `medvoice-app`) so it can roll back.
3. Exports `MEDVOICE_IMAGE` (registry image + tag) and `MEDVOICE_ENV_FILE` (server env-file path), then runs `docker compose pull medvoice-app` and `docker compose up -d --no-build`. The `--no-build` flag guarantees the server never builds; it only runs the pulled image.
4. Waits for the compose healthcheck (liveness only, `GET /health/live`) to report `healthy`.
5. Runs the smoke test: `GET /health/live` must return `"status":"alive"` and `GET /health` must return HTTP 200 with `"status":"ok"` or `"status":"degraded"`.
6. On failure, redeploys the recorded previous image and fails the job.

> **Note (A4.3.1):** the smoke test calls `GET /health`, which probes
> Foundry, Munsit and Voxtral (cached 30 s). Only **Foundry** is a hard
> dependency: `/health` returns 503 only when Foundry is unreachable or its
> key/token is rejected. A down Munsit/Voxtral is reported as `degraded` and
> keeps HTTP 200, so an STT outage does **not** fail a deploy.
> `/health/live` remains the dependency-free liveness probe.

## Parameters and variables

- `deployStaging` and `deployProduction` default to `false`; production depends on staging.
- Environments: `dev`, `staging`, and `production`.
- Placeholder variable groups: `<PLACEHOLDER_DEV_VARIABLE_GROUP>`, `<PLACEHOLDER_STAGING_VARIABLE_GROUP>`, and `<PLACEHOLDER_PRODUCTION_VARIABLE_GROUP>`. Each must provide `MEDVOICE_ENV_FILE` (secret) and `MEDVOICE_SMOKE_TEST_BASE_URL`.
- Per-stage variables: `MEDVOICE_COMPOSE_DIR` (`<PLACEHOLDER_DEV_COMPOSE_DIR>`, `<PLACEHOLDER_STAGING_COMPOSE_DIR>`, `<PLACEHOLDER_PRODUCTION_COMPOSE_DIR>`) — the path to the project checkout on the target server that contains `docker-compose.yml`.

## Azure DevOps setup, in order

1. Register `<PLACEHOLDER_SELF_HOSTED_AGENT_POOL>`. Deploy agents need Docker with the Compose v2 plugin, `curl`, registry access, and network access to the target's `/health/live`.
2. Create `<PLACEHOLDER_REGISTRY_SERVICE_CONNECTION>` with registry push/pull rights; replace `<PLACEHOLDER_REGISTRY_REPOSITORY>` and pin `<PLACEHOLDER_RUFF_VERSION>` to an approved version.
3. Create Environments `dev`, `staging`, and `production`, and attach the appropriate deploy targets/agents.
4. Create and restrict the three variable groups. Populate the pre-provisioned server env-file path, the private smoke-test base URL, and the compose directory. No server path or port is defined in this repository.
5. In **Pipelines > Environments > production > Approvals and checks**, add the required manual approvers. The production job targets that exact environment, so the approval executes before deployment.
6. Add pipeline build validation as a required branch policy on `main`, then complete a green main run.

## What the target server must have

- Docker Engine and the Compose v2 plugin (`docker compose version` works).
- The project checkout containing `docker-compose.yml` at `MEDVOICE_COMPOSE_DIR`.
- The environment file at `MEDVOICE_ENV_FILE`, pre-provisioned and readable by the deploy user. It is never created or printed by the pipeline.
- Host folders for the writable volumes (`src/uploads`, `src/logs`) that are git/docker-ignored.
- Network access to the registry and to the app's `/health/live` endpoint.

## Manual/infrastructure work

This change creates no Azure resources, agents, registry connection, target host, variable groups, approval, or networking. Provision staging with the same deployment topology/configuration contract as production. Private-network-only access is infrastructure/APIM work dependent on A2.3.

# CI/CD and staging (A4)

`azure-pipelines.yml` runs on `main` and pull requests to `main`. CI installs `src/requirements.txt` plus `requirements-test.txt`, checks Ruff runtime-error rules, then runs concurrency and full tests with inline dummy Foundry configuration. It never reads or publishes `.env`. `tests/test_m1_verification.py` is explicitly ignored because it loads `src/.env` and calls live Azure infrastructure. A successful main build uses the existing `Dockerfile` and pushes Build ID and short SHA tags.

All dev, staging, and production deployments use `pipelines/deploy-template.yml`; only the environment name and Azure DevOps variable group differ. The template pulls the requested tag, records the existing image, starts with the pre-existing target env file, retries `GET /health/live` (liveness) and `GET /health` (dependency health) ten times, and restores the old image if smoke testing fails. It never creates or prints the env file.

> **Note (A4.3.1):** the smoke test now also calls `GET /health`, which probes
> Foundry, Munsit and Voxtral (cached 30 s). Only **Foundry** is a hard
> dependency: `/health` returns 503 only when Foundry is unreachable or its
> key/token is rejected. A down Munsit/Voxtral is reported as `degraded` and
> keeps HTTP 200, so an STT outage does **not** fail a deploy.
> `/health/live` remains the dependency-free liveness probe.

## Parameters and variables

- `deployStaging` and `deployProduction` default to `false`; production depends on staging.
- Environments: `dev`, `staging`, and `production`.
- Placeholder variable groups: `<PLACEHOLDER_DEV_VARIABLE_GROUP>`, `<PLACEHOLDER_STAGING_VARIABLE_GROUP>`, and `<PLACEHOLDER_PRODUCTION_VARIABLE_GROUP>`. Each must provide `MEDVOICE_ENV_FILE` (secret) and `MEDVOICE_SMOKE_TEST_BASE_URL`.

## Azure DevOps setup, in order

1. Register `<PLACEHOLDER_SELF_HOSTED_AGENT_POOL>`. Deploy agents need Docker, `curl`, registry access, and network access to the target's `/health/live`.
2. Create `<PLACEHOLDER_REGISTRY_SERVICE_CONNECTION>` with registry push/pull rights; replace `<PLACEHOLDER_REGISTRY_REPOSITORY>` and pin `<PLACEHOLDER_RUFF_VERSION>` to an approved version.
3. Create Environments `dev`, `staging`, and `production`, and attach the appropriate deploy targets/agents.
4. Create and restrict the three variable groups. Populate the pre-provisioned server env-file path and private smoke-test base URL. Replace `<PLACEHOLDER_HOST_PORT>` with the host port; no server path or port is defined in this repository.
5. In **Pipelines > Environments > production > Approvals and checks**, add the required manual approvers. The production job targets that exact environment, so the approval executes before deployment.
6. Add pipeline build validation as a required branch policy on `main`, then complete a green main run.

## Manual/infrastructure work

This change creates no Azure resources, agents, registry connection, target host, variable groups, approval, or networking. Provision staging with the same deployment topology/configuration contract as production. Private-network-only access is infrastructure/APIM work dependent on A2.3.

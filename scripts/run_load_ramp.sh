#!/usr/bin/env bash
# A6.1.1: run the 1 -> 5 -> 10 -> 20 -> 50 concurrent-session ramp.
#
# CREATE ONLY — the owner runs this against a staging environment.
# 10 minutes per step, per language, saving a Locust report per step.
#
# Usage:
#   LOCUST_HOST=http://staging-host:7777 ./scripts/run_load_ramp.sh
#
# Environment:
#   LOCUST_HOST        Target base URL (required)
#   LOCUST_OUT_DIR     Output directory (default: reports/load_ramp)
#   LOCUST_STEP_SECONDS  Seconds per step (default: 600 = 10 min)
#   LOCUST_LANGUAGES   Space-separated languages (default: "en ar")
set -euo pipefail

HOST="${LOCUST_HOST:?Set LOCUST_HOST to the staging base URL}"
OUT_DIR="${LOCUST_OUT_DIR:-reports/load_ramp}"
STEP_SECONDS="${LOCUST_STEP_SECONDS:-600}"
LANGUAGES="${LOCUST_LANGUAGES:-en ar}"
STEPS=(1 5 10 20 50)

mkdir -p "$OUT_DIR"

for lang in $LANGUAGES; do
  for users in "${STEPS[@]}"; do
    tag="${lang}_u${users}"
    echo "=== Ramp step: language=${lang} users=${users} for ${STEP_SECONDS}s ==="
    LOCUST_LANGUAGE="$lang" \
    LOCUST_SESSION_CSV="${OUT_DIR}/sessions_${tag}.csv" \
    locust -f locustfile.py \
      --host "$HOST" \
      --headless \
      -u "$users" -r "$users" \
      -t "${STEP_SECONDS}s" \
      --csv "${OUT_DIR}/locust_${tag}" \
      --html "${OUT_DIR}/locust_${tag}.html"
  done
done

echo "Ramp complete. Reports in ${OUT_DIR}."
echo "Fill docs/capacity_report_template.md from the per-step CSVs."
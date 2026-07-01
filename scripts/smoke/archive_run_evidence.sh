#!/usr/bin/env bash
# Pull the mechanical evidence a smoke run wrote to remote scratch back onto
# the controller, into runs/<slug>/evidence/. Without this, a run's proof
# (validation_summary.json's status field, per the rider's own evidence bar)
# lives only on ephemeral Spark-host scratch and is lost the moment that
# host's disk gets reused -- the smoke script's only cross-host data movement
# is a one-way push (controller/head -> worker), nothing pulls results back.
#
# Usage:
#   scripts/smoke/archive_run_evidence.sh <host_ssh_target> <remote_run_dir> <local_slug>
#
# Example:
#   scripts/smoke/archive_run_evidence.sh spark \
#     /home/jarrodbarnes/nemo-rl-smoke/nemo-polar-qwen3-0p6b-<slug> <slug>
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <host_ssh_target> <remote_run_dir> <local_slug>" >&2
  exit 1
fi

HOST_SSH="$1"
REMOTE_RUN_DIR="$2"
LOCAL_SLUG="$3"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST_DIR="${REPO_ROOT}/runs/${LOCAL_SLUG}/evidence"
mkdir -p "${DEST_DIR}"

EVIDENCE_FILES=(
  validation_summary.json
  metrics.jsonl
  run_timing.json
  train_data_audit.json
)

pulled=0
missing=()
for f in "${EVIDENCE_FILES[@]}"; do
  if ssh "${HOST_SSH}" "test -f '${REMOTE_RUN_DIR}/${f}'" 2>/dev/null; then
    scp -q "${HOST_SSH}:${REMOTE_RUN_DIR}/${f}" "${DEST_DIR}/${f}"
    pulled=$((pulled + 1))
  else
    missing+=("${f}")
  fi
done

# Also pull the raw logs directory (exit-code.log, nohup log if present,
# grpo-polar step logs) so a failed run's partial evidence is preserved too.
if ssh "${HOST_SSH}" "test -d '${REMOTE_RUN_DIR}/logs'" 2>/dev/null; then
  rsync -az "${HOST_SSH}:${REMOTE_RUN_DIR}/logs/" "${DEST_DIR}/logs/" 2>/dev/null || true
fi

echo "pulled ${pulled}/${#EVIDENCE_FILES[@]} evidence files to ${DEST_DIR}"
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "missing (not found on remote host): ${missing[*]}"
fi

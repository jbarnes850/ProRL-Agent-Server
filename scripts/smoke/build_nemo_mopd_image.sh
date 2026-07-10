#!/usr/bin/env bash
set -euo pipefail

# Build a separate NeMo RL image from an upstream ref with GRPO-native MOPD.
# Observed upstream main on 2026-07-08.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NEMO_RL_REF="${NEMO_RL_REF:-1687d23347f176a8f19dfe84804c78cc6650cdcc}"
export IMAGE="${IMAGE:-local/nemo-rl-main-cu132:${NEMO_RL_REF:0:8}}"
export REQUIRE_MOPD="${REQUIRE_MOPD:-1}"

exec "${SCRIPT_DIR}/build_nemo_async_image.sh"

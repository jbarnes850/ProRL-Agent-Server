#!/usr/bin/env bash
set -euo pipefail

# Build the NeMo RL release image from a pinned upstream revision via NeMo's own Dockerfile: one coherent CUDA/Torch/vLLM/Ray env, then local import/GPU gates.

NEMO_RL_REF="${NEMO_RL_REF:-37526dfac0a80b7032659a3ea030e0a9f69f99c6}"
IMAGE="${IMAGE:-local/nemo-rl-main-cu132:${NEMO_RL_REF:0:8}}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/cuda-dl-base:26.03-cuda13.2-devel-ubuntu24.04}"
BUILD_CONTEXT="${BUILD_CONTEXT:-https://github.com/NVIDIA-NeMo/RL.git#${NEMO_RL_REF}}"
BUILD_LOG="${BUILD_LOG:-}"

build_cmd=(
  docker buildx build
  --progress=plain
  --target release
  --build-arg "BASE_IMAGE=${BASE_IMAGE}"
  --build-arg "NRL_GIT_REF=${NEMO_RL_REF}"
  --build-arg "NEMO_RL_COMMIT=${NEMO_RL_REF}"
  --build-arg "NVIDIA_BUILD_REF=${NEMO_RL_REF}"
  -f docker/Dockerfile
  -t "${IMAGE}"
  "${BUILD_CONTEXT}"
)

echo "building ${IMAGE}"
echo "nemo_ref=${NEMO_RL_REF}"
echo "base_image=${BASE_IMAGE}"
echo "context=${BUILD_CONTEXT}"

if [[ -n "${BUILD_LOG}" ]]; then
  mkdir -p "$(dirname "${BUILD_LOG}")"
  "${build_cmd[@]}" 2>&1 | tee "${BUILD_LOG}"
else
  "${build_cmd[@]}"
fi

docker run --rm --gpus all --ipc=host "${IMAGE}" bash -lc 'python3 - <<'"'"'PY'"'"'
import importlib
import os
import platform

import torch

required = [
    "nemo_rl.algorithms.grpo",
    "nemo_rl.algorithms.async_utils.trajectory_collector",
    "nemo_rl.algorithms.async_utils.replay_buffer",
    "nemo_rl.models.generation.vllm.vllm_worker",
    "nemo_rl.models.generation.vllm.vllm_worker_async",
    "tensordict",
]

print("platform", platform.machine())
print("nemo_ref", os.environ.get("NEMO_RL_COMMIT"))
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the release image")

print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
print("arch_list", torch.cuda.get_arch_list())
if torch.cuda.get_device_capability(0) < (12, 1):
    raise SystemExit("expected GB10 capability >= 12.1")
if not torch.__version__.startswith("2.11.0"):
    raise SystemExit(f"expected torch 2.11.0, got {torch.__version__}")
if torch.version.cuda is None or not torch.version.cuda.startswith("13."):
    raise SystemExit(f"expected CUDA 13 torch wheel, got {torch.version.cuda}")

for module in required:
    imported = importlib.import_module(module)
    print(module, getattr(imported, "__file__", "built-in"))

print("release image gpu/import gate ok")
PY'

docker run --rm "${IMAGE}" bash -lc 'set -euo pipefail
for py in \
  /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python \
  /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python \
  /opt/ray_venvs/nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector/bin/python \
  /opt/ray_venvs/nemo_rl.algorithms.async_utils.ReplayBuffer/bin/python \
  /opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python; do
  if [ ! -x "$py" ]; then
    echo "missing actor python: $py" >&2
    exit 1
  fi
  expect_vllm=0
  case "$py" in
    *vllm*|*Vllm*) expect_vllm=1 ;;
  esac
  EXPECT_VLLM="$expect_vllm" "$py" - <<'"'"'PY'"'"'
import importlib
import os
import sys

modules = ["torch", "nemo_rl", "tensordict"]
if os.environ.get("EXPECT_VLLM") == "1":
    modules.append("vllm")

for module in modules:
    imported = importlib.import_module(module)
    print(sys.executable, module, getattr(imported, "__file__", "built-in"))
PY
done'

echo "built ${IMAGE} from NVIDIA-NeMo/RL@${NEMO_RL_REF}"

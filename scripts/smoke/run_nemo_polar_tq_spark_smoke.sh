#!/usr/bin/env bash
set -euo pipefail

NEMO_RL_REF="${NEMO_RL_REF:-37526dfac0a80b7032659a3ea030e0a9f69f99c6}"
IMAGE="${IMAGE:-local/nemo-rl-main-cu132:${NEMO_RL_REF:0:8}}"
HEAD_IP="${HEAD_IP:-192.168.100.10}"
WORKER_IP="${WORKER_IP:-192.168.100.11}"
WORKER_SSH="${WORKER_SSH:-jarrodbarnes@192.168.100.11}"
POLAR_ROOT="${POLAR_ROOT:-/home/jarrodbarnes/polar-nemo-lab}"
POLAR_PYTHON="${POLAR_PYTHON:-${POLAR_ROOT}/.venv/bin/python}"
MODEL_HOST="${MODEL_HOST:-/home/jarrodbarnes/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
MODEL_CONT="${MODEL_CONT:-/host-hf/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
STAMP="${1:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/home/jarrodbarnes/nemo-rl-smoke/nemo-polar-tq-qwen3-0p6b-${STAMP}}"
HEAD_CONTAINER="nemo-polar-tq-head-${STAMP}"
WORKER_CONTAINER="nemo-polar-tq-worker-${STAMP}"
POLAR_ROLLOUT_URL="http://${WORKER_IP}:18080"
POLAR_GATEWAY_URL="http://${WORKER_IP}:18100"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://${WORKER_IP}:30000/v1}"

echo "This training run is worth doing because it will improve the go/no-go decision for Jarrod's two-Spark RL lab as measured by a one-step live Polar-to-NeMo GRPO smoke, producing a run artifact that proves or rejects the rollout, reward, TransferQueue, and weight-sync path."
echo "run_dir=${RUN_DIR}"

mkdir -p "${RUN_DIR}"/{data,logs,config,ray-head,tmp,hf}
ssh "${WORKER_SSH}" "mkdir -p '${RUN_DIR}'/{data,logs,config,ray-worker,tmp,hf,polar-results}"

cat > "${RUN_DIR}/data/train.jsonl" <<'JSONL'
{"input":"Answer with only the digit: what is 1 + 1?","output":"2"}
{"input":"Answer with only the digit: what is 2 + 2?","output":"4"}
{"input":"Answer with only the digit: what is 3 + 5?","output":"8"}
{"input":"Answer with only the digit: what is 10 - 7?","output":"3"}
JSONL
scp -q "${RUN_DIR}/data/train.jsonl" "${WORKER_SSH}:${RUN_DIR}/data/train.jsonl"

cat > "${RUN_DIR}/config/topology.yaml" <<YAML
rollout:
  host: 0.0.0.0
  port: 18080
  public_url: ${POLAR_ROLLOUT_URL}
  save_dir: ${RUN_DIR}/polar-results
  dispatch_poll_interval_seconds: 0.25
  callback_grace_seconds: 120

gateway:
  rollout_server_url: ${POLAR_ROLLOUT_URL}
  heartbeat_interval_seconds: 5
  nodes:
    - id: cfd0-vllm
      host: 0.0.0.0
      port: 18100
      public_url: ${POLAR_GATEWAY_URL}
      max_init_workers: 4
      max_run_workers: 4
      max_postrun_workers: 4
      model_served: ${MODEL_CONT}
      inference:
        engine: vllm
        base_url: ${VLLM_BASE_URL}
      default_runtime:
        backend: docker
        image: ${IMAGE}
        network: host
        workdir: /polar/session
        cpus: 2
        memory_mb: 4096
        gpus: 0
YAML
scp -q "${RUN_DIR}/config/topology.yaml" "${WORKER_SSH}:${RUN_DIR}/config/topology.yaml"

docker rm -f "${HEAD_CONTAINER}" >/dev/null 2>&1 || true
ssh "${WORKER_SSH}" "docker rm -f '${WORKER_CONTAINER}' >/dev/null 2>&1 || true"
ssh "${WORKER_SSH}" "if [ -f '${RUN_DIR}/polar-rollout.pid' ]; then kill \$(cat '${RUN_DIR}/polar-rollout.pid') >/dev/null 2>&1 || true; fi; if [ -f '${RUN_DIR}/polar-gateway.pid' ]; then kill \$(cat '${RUN_DIR}/polar-gateway.pid') >/dev/null 2>&1 || true; fi"

COMMON_ENV=(
  -e NVIDIA_VISIBLE_DEVICES=all
  -e CUDA_VISIBLE_DEVICES=0
  -e NCCL_IB_DISABLE=1
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1
  -e GLOO_SOCKET_IFNAME=enp1s0f1np1
  -e NRL_REFIT_BUFFER_MEMORY_RATIO=0.05
  -e PYTHONUNBUFFERED=1
  -e PYTHONPATH=/work/polar/src:/opt/nemo-rl
  -e HF_HOME=/work/hf
  -e UV_CACHE_DIR=/work/uv-cache
  -e VLLM_CACHE_ROOT=/work/vllm-cache
  -e NEMO_RL_POLAR_ROLLOUT=1
  -e NEMO_RL_NVML_MEM_GET_INFO_FALLBACK=1
)

COMMON_DOCKER=(
  --gpus all
  --network host
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  --shm-size=32g
  -v "${MODEL_HOST%/hub/*}/hub:/host-hf/hub:ro"
  -v "${POLAR_ROOT}:/work/polar:ro"
)

echo "starting Ray head container on ${HEAD_IP}"
docker run -d --name "${HEAD_CONTAINER}" \
  "${COMMON_DOCKER[@]}" \
  -v "${RUN_DIR}:/work" \
  "${COMMON_ENV[@]}" \
  "${IMAGE}" \
  bash -lc "ray stop --force >/dev/null 2>&1 || true; ray start --head --node-ip-address=${HEAD_IP} --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --dashboard-agent-listen-port=52365 --dashboard-agent-grpc-port=53007 --runtime-env-agent-port=53005 --node-manager-port=53001 --object-manager-port=53003 --metrics-export-port=53009 --min-worker-port=54001 --max-worker-port=54257 --num-gpus=1 --num-cpus=16 --disable-usage-stats --block" \
  > "${RUN_DIR}/head.container.id"

sleep 8
docker logs "${HEAD_CONTAINER}" --tail 80 | tee "${RUN_DIR}/logs/head-start.log"

echo "starting Ray worker container on ${WORKER_IP}"
ssh "${WORKER_SSH}" bash -s <<EOF
set -euo pipefail
docker run -d --name "${WORKER_CONTAINER}" \
  --gpus all --network host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=32g \
  -v "${MODEL_HOST%/hub/*}/hub:/host-hf/hub:ro" \
  -v "${POLAR_ROOT}:/work/polar:ro" \
  -v "${RUN_DIR}:/work" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1 \
  -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
  -e NRL_REFIT_BUFFER_MEMORY_RATIO=0.05 \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/work/polar/src:/opt/nemo-rl \
  -e HF_HOME=/work/hf \
  -e UV_CACHE_DIR=/work/uv-cache \
  -e VLLM_CACHE_ROOT=/work/vllm-cache \
  -e NEMO_RL_POLAR_ROLLOUT=1 \
  -e NEMO_RL_NVML_MEM_GET_INFO_FALLBACK=1 \
  "${IMAGE}" \
  bash -lc "ray stop --force >/dev/null 2>&1 || true; ray start --address=${HEAD_IP}:6379 --node-ip-address=${WORKER_IP} --dashboard-agent-listen-port=52365 --dashboard-agent-grpc-port=53007 --runtime-env-agent-port=53005 --node-manager-port=53001 --object-manager-port=53003 --metrics-export-port=53009 --min-worker-port=54001 --max-worker-port=54257 --num-gpus=1 --num-cpus=16 --disable-usage-stats --block" \
  > "${RUN_DIR}/worker.container.id"
EOF

sleep 12
ssh "${WORKER_SSH}" "docker logs '${WORKER_CONTAINER}' --tail 80" | tee "${RUN_DIR}/logs/worker-start.log"

echo "patching NeMo in both containers"
docker exec "${HEAD_CONTAINER}" bash -lc "cd /opt/nemo-rl && /work/polar/scripts/patch/patch_nemo_polar.sh" | tee "${RUN_DIR}/logs/patch-head.log"
ssh "${WORKER_SSH}" "docker exec '${WORKER_CONTAINER}' bash -lc 'cd /opt/nemo-rl && /work/polar/scripts/patch/patch_nemo_polar.sh'" | tee "${RUN_DIR}/logs/patch-worker.log"

echo "verifying imports and patch contracts"
verify_cmd='cd /opt/nemo-rl && python3 - <<'"'"'PY'"'"'
import importlib
from pathlib import Path

for module in [
    "nemo_rl.algorithms.grpo_sync",
    "nemo_rl.data_plane.adapters.transfer_queue",
    "nemo_bridge.rollout_actor",
    "transfer_queue",
    "tensordict",
]:
    imported = importlib.import_module(module)
    print(module, getattr(imported, "__file__", "built-in"))
text = Path("nemo_rl/algorithms/grpo_sync.py").read_text()
assert "nemo_bridge.rollout_actor" in text
assert "loss_multiplier" in text
print("patch contract ok")
PY'
docker exec "${HEAD_CONTAINER}" bash -lc "${verify_cmd}" | tee "${RUN_DIR}/logs/import-head.log"
ssh "${WORKER_SSH}" "docker exec '${WORKER_CONTAINER}' bash -lc $(printf '%q' "${verify_cmd}")" | tee "${RUN_DIR}/logs/import-worker.log"

echo "verifying Ray cluster resources"
docker exec "${HEAD_CONTAINER}" bash -lc "RAY_ADDRESS=${HEAD_IP}:6379 python - <<'PY'
import json
import ray

ray.init(address='auto')
resources = ray.cluster_resources()
nodes = ray.nodes()
print(json.dumps(resources, sort_keys=True))
print(json.dumps(nodes, default=str, sort_keys=True))
gpu_count = int(resources.get('GPU', 0))
alive_nodes = [node for node in nodes if node.get('Alive')]
if gpu_count < 2 or len(alive_nodes) < 2:
    raise SystemExit(f'expected >=2 GPUs and >=2 alive nodes, got resources={resources}, alive_nodes={len(alive_nodes)}')
ray.shutdown()
PY" | tee "${RUN_DIR}/logs/ray-resources.log"

echo "starting Polar rollout and gateway on ${WORKER_IP}"
ssh "${WORKER_SSH}" "(cd '${POLAR_ROOT}' && exec env PYTHONPATH=src '${POLAR_PYTHON}' -m polar.cli serve_rollout -c '${RUN_DIR}/config/topology.yaml') </dev/null > '${RUN_DIR}/logs/polar-rollout.log' 2>&1 & echo \$! > '${RUN_DIR}/polar-rollout.pid'; exit 0"
sleep 3
ssh "${WORKER_SSH}" "(cd '${POLAR_ROOT}' && exec env PYTHONPATH=src '${POLAR_PYTHON}' -m polar.cli serve_gateway -c '${RUN_DIR}/config/topology.yaml' --node-id cfd0-vllm) </dev/null > '${RUN_DIR}/logs/polar-gateway.log' 2>&1 & echo \$! > '${RUN_DIR}/polar-gateway.pid'; exit 0"

for _ in $(seq 1 60); do
  if curl -fsS "${POLAR_ROLLOUT_URL}/health" >/dev/null && curl -fsS "${POLAR_GATEWAY_URL}/health" >/dev/null; then
    break
  fi
  sleep 1
done
curl -fsS "${POLAR_ROLLOUT_URL}/health" | tee "${RUN_DIR}/logs/polar-rollout-health.json"
curl -fsS "${POLAR_GATEWAY_URL}/health" | tee "${RUN_DIR}/logs/polar-gateway-health.json"
curl -fsS "${POLAR_ROLLOUT_URL}/rollout/status" | tee "${RUN_DIR}/logs/polar-status-before.json"

echo "running Polar-backed TransferQueue GRPO smoke"
set +e
docker exec \
  -e RAY_ADDRESS="${HEAD_IP}:6379" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1 \
  -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
  -e NRL_REFIT_BUFFER_MEMORY_RATIO=0.05 \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/work/polar/src:/opt/nemo-rl \
  -e NEMO_RL_POLAR_ROLLOUT=1 \
  -e NEMO_RL_NVML_MEM_GET_INFO_FALLBACK=1 \
  "${HEAD_CONTAINER}" bash -lc "
    cd /opt/nemo-rl
    python examples/run_grpo.py \
      --config examples/configs/recipes/llm/grpo-qwen3-0.6b-1n8g-sglang.yaml \
      grpo.num_prompts_per_step=1 \
      grpo.num_generations_per_prompt=4 \
      grpo.max_num_steps=1 \
      grpo.max_num_epochs=1 \
      grpo.val_at_start=false \
      grpo.val_at_end=false \
      grpo.val_period=9999 \
      policy.model_name=${MODEL_CONT} \
      policy.tokenizer.name=${MODEL_CONT} \
      +policy.dtensor_cfg.automodel_kwargs.force_hf=true \
      policy.train_global_batch_size=4 \
      policy.train_micro_batch_size=1 \
      policy.logprob_batch_size=1 \
      policy.generation_batch_size=1 \
      policy.max_total_sequence_length=256 \
      policy.sequence_packing.enabled=false \
      policy.dynamic_batching.enabled=false \
      policy.generation.backend=vllm \
      policy.generation.max_new_tokens=8 \
      policy.generation.port_range_low=30000 \
      policy.generation.port_range_high=30001 \
      policy.generation.vllm_cfg.async_engine=true \
      +policy.generation.vllm_cfg.expose_http_server=true \
      policy.generation.vllm_cfg.enable_vllm_metrics_logger=false \
      policy.generation.vllm_cfg.max_model_len=256 \
      policy.generation.vllm_cfg.gpu_memory_utilization=0.35 \
      policy.generation.vllm_cfg.enforce_eager=true \
      policy.generation.colocated.enabled=false \
      policy.generation.colocated.resources.gpus_per_node=1 \
      policy.generation.colocated.resources.num_nodes=1 \
      data.train.dataset_name=ResponseDataset \
      +data.train.data_path=/work/data/train.jsonl \
      +data.train.input_key=input \
      +data.train.output_key=output \
      data.train.split_validation_size=0.5 \
      data.default.processor=math_hf_data_processor \
      data.default.env_name=math \
      env.math.num_workers=1 \
      data_plane.enabled=true \
      data_plane.impl=transfer_queue \
      data_plane.backend=simple \
      data_plane.storage_capacity=1000000 \
      data_plane.num_storage_units=2 \
      logger.log_dir=/work/logs/grpo \
      logger.wandb_enabled=false \
      logger.tensorboard_enabled=false \
      logger.mlflow_enabled=false \
      logger.swanlab_enabled=false \
      logger.monitor_gpus=false \
      checkpointing.enabled=false \
      cluster.gpus_per_node=1 \
      cluster.num_nodes=2 \
      +polar.rollout_url=${POLAR_ROLLOUT_URL} \
      +polar.task_id_prefix=nemo-polar-qwen3-${STAMP} \
      +polar.model_name=${MODEL_CONT} \
      +polar.max_tokens=8 \
      +polar.temperature=1.0 \
      +polar.top_p=1.0 \
      +polar.session_timeout_seconds=240 \
      +polar.task_timeout_seconds=360 \
      +polar.poll_interval_seconds=1 \
      +polar.request_timeout_seconds=180 \
      +polar.trace_selection=last \
      +polar.max_policy_staleness=0 \
      +polar.require_reward_variance=false \
      +polar.runtime.backend=docker \
      +polar.runtime.image=${IMAGE} \
      +polar.runtime.network=host \
      +polar.runtime.workdir=/polar/session \
      +polar.runtime.gpus=0 \
      +polar.runtime.cpus=2 \
      +polar.runtime.memory_mb=4096 \
      +polar.builder.strategy=per_request \
      +polar.evaluator.strategy=response_match \
      '+polar.evaluator.config.patterns=[\"2\",\"3\",\"4\",\"8\"]' \
      +polar.evaluator.config.use_regex=false
  " 2>&1 | tee "${RUN_DIR}/logs/grpo-polar-tq.log"
code=${PIPESTATUS[0]}
set -e
echo "docker_exec_exit_code=${code}" | tee "${RUN_DIR}/logs/exit-code.log"

curl -fsS "${POLAR_ROLLOUT_URL}/rollout/status" | tee "${RUN_DIR}/logs/polar-status-after.json" || true
ssh "${WORKER_SSH}" "find '${RUN_DIR}/polar-results' -type f | sort" | tee "${RUN_DIR}/logs/polar-result-files.log" || true
ssh "${WORKER_SSH}" "find '${RUN_DIR}/polar-results' -type f -name '*.json' -o -name '*.jsonl' | sort | head -20 | xargs -r -n1 sh -c 'echo --- \$1; python3 -m json.tool \"\$1\" 2>/dev/null | head -120 || head -120 \"\$1\"' sh" > "${RUN_DIR}/logs/polar-result-preview.log" 2>&1 || true

echo "collecting final logs"
docker logs "${HEAD_CONTAINER}" --tail 160 > "${RUN_DIR}/logs/head-final.log" 2>&1 || true
ssh "${WORKER_SSH}" "docker logs '${WORKER_CONTAINER}' --tail 160" > "${RUN_DIR}/logs/worker-final.log" 2>&1 || true
ssh "${WORKER_SSH}" "tail -160 '${RUN_DIR}/logs/polar-rollout.log'; echo ---GATEWAY---; tail -200 '${RUN_DIR}/logs/polar-gateway.log'" > "${RUN_DIR}/logs/polar-final.log" 2>&1 || true

if [[ "${code}" -eq 0 ]] \
  && grep -q "Running synchronous GRPO training (TransferQueue)" "${RUN_DIR}/logs/grpo-polar-tq.log" \
  && grep -q "PolarSyncRolloutActor: wrote" "${RUN_DIR}/logs/grpo-polar-tq.log" \
  && grep -q "transfer_and_update_weights" "${RUN_DIR}/logs/grpo-polar-tq.log"; then
  printf 'SUCCEEDED %s\n' "$(date -Is)" | tee "${RUN_DIR}/status"
else
  printf 'FAILED code=%s %s\n' "${code}" "$(date -Is)" | tee "${RUN_DIR}/status"
fi

if [[ "${KEEP_CONTAINERS:-0}" != "1" ]]; then
  docker rm -f "${HEAD_CONTAINER}" >/dev/null 2>&1 || true
  ssh "${WORKER_SSH}" "docker rm -f '${WORKER_CONTAINER}' >/dev/null 2>&1 || true"
fi
if [[ "${KEEP_POLAR:-0}" != "1" ]]; then
  ssh "${WORKER_SSH}" "kill \$(cat '${RUN_DIR}/polar-rollout.pid') >/dev/null 2>&1 || true; kill \$(cat '${RUN_DIR}/polar-gateway.pid') >/dev/null 2>&1 || true"
fi

exit "${code}"

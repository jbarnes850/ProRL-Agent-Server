#!/usr/bin/env bash
set -euo pipefail

# See the matching comment in run_nemo_polar_external_collector_spark_smoke.sh:
# c236061b is the proven, currently-deployed revision; the prior default
# predates CISPO's use_cispo config key and silently produces a Hydra
# struct error for any cispo-algorithm spec that relies on this default.
NEMO_RL_REF="${NEMO_RL_REF:-c236061b250e97638722292ab8a54d5eb47ae00f}"
IMAGE="${IMAGE:-local/nemo-rl-main-cu132:${NEMO_RL_REF:0:8}}"
HEAD_IP="${HEAD_IP:-192.168.100.10}"
WORKER_IP="${WORKER_IP:-192.168.100.11}"
WORKER_SSH="${WORKER_SSH:-jarrodbarnes@192.168.100.11}"
HEAD_HOSTNAME="${HEAD_HOSTNAME:-spark-f7e2}"
WORKER_HOSTNAME="${WORKER_HOSTNAME:-spark-cfd0}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}"
# See the matching comment in run_nemo_polar_external_collector_spark_smoke.sh:
# Ray's uncapped default object-store reservation (~30% of node memory)
# competes with model weights and the offload/onload cycle on DGX Spark's
# unified memory pool. This topology moves weights via NCCL broadcast, not
# Ray's plasma store, so capping this recovers real headroom.
RAY_OBJECT_STORE_MEMORY_BYTES="${RAY_OBJECT_STORE_MEMORY_BYTES:-8589934592}"
MODEL_HOST="${MODEL_HOST:-/home/jarrodbarnes/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
MODEL_CONT="${MODEL_CONT:-/host-hf/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
STAMP="${1:-$(date +%Y%m%d-%H%M%S)}"
shift || true
EXTRA_OVERRIDES=("$@")
RUN_DIR="${RUN_DIR:-/home/jarrodbarnes/nemo-rl-smoke/nemo-async-qwen3-0p6b-${STAMP}}"
HEAD_CONTAINER="nemo-async-head-${STAMP}"
WORKER_CONTAINER="nemo-async-worker-${STAMP}"

echo "This training run is worth doing because it will improve the go/no-go decision for Jarrod's two-Spark RL lab as measured by a one-step live NeMo native Async GRPO smoke, producing a run artifact that proves or rejects native async replay-buffer, staleness, IS-correction, and weight-sync behavior before Polar integration."
echo "run_dir=${RUN_DIR}"
echo "image=${IMAGE}"

mkdir -p "${RUN_DIR}"/{data,logs,ray-head,tmp,hf}
ssh "${WORKER_SSH}" "mkdir -p '${RUN_DIR}'/{data,logs,ray-worker,tmp,hf}"

cat > "${RUN_DIR}/nccl.conf" <<EOF
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}
NCCL_SOCKET_FAMILY=AF_INET
NCCL_NET=Socket
NCCL_NET_PLUGIN=none
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET,ENV
EOF
scp -q "${RUN_DIR}/nccl.conf" "${WORKER_SSH}:${RUN_DIR}/nccl.conf"

cat > "${RUN_DIR}/data/train.jsonl" <<'JSONL'
{"input":"Answer with only the digit: what is 1 + 1?","output":"2"}
{"input":"Answer with only the digit: what is 2 + 2?","output":"4"}
{"input":"Answer with only the digit: what is 3 + 5?","output":"8"}
{"input":"Answer with only the digit: what is 10 - 7?","output":"3"}
JSONL
scp -q "${RUN_DIR}/data/train.jsonl" "${WORKER_SSH}:${RUN_DIR}/data/train.jsonl"

docker rm -f "${HEAD_CONTAINER}" >/dev/null 2>&1 || true
ssh "${WORKER_SSH}" "docker rm -f '${WORKER_CONTAINER}' >/dev/null 2>&1 || true"

COMMON_ENV=(
  -e NVIDIA_VISIBLE_DEVICES=all
  -e CUDA_VISIBLE_DEVICES=0
  -e NCCL_IB_DISABLE=1
  -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}
  -e NCCL_SOCKET_FAMILY=AF_INET
  -e NCCL_NET=Socket
  -e NCCL_NET_PLUGIN=none
  -e NCCL_DEBUG=INFO
  -e NCCL_DEBUG_SUBSYS=INIT,NET,ENV
  -e GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}
  -e NRL_REFIT_BUFFER_MEMORY_RATIO=0.05
  -e PYTHONUNBUFFERED=1
  -e HF_HOME=/work/hf
  -e UV_CACHE_DIR=/work/uv-cache
  -e VLLM_CACHE_ROOT=/work/vllm-cache
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
  -v "${RUN_DIR}/nccl.conf:/etc/nccl.conf:ro"
)

start_worker_container() {
  echo "starting Ray worker container on ${WORKER_IP}"
  ssh "${WORKER_SSH}" bash -s <<EOF
set -euo pipefail
docker run -d --name "${WORKER_CONTAINER}" \
  --gpus all --network host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=32g \
  -v "${MODEL_HOST%/hub/*}/hub:/host-hf/hub:ro" \
  -v "${RUN_DIR}:/work" \
  -v "${RUN_DIR}/nccl.conf:/etc/nccl.conf:ro" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
  -e NCCL_SOCKET_FAMILY=AF_INET \
  -e NCCL_NET=Socket \
  -e NCCL_NET_PLUGIN=none \
  -e NCCL_DEBUG=INFO \
  -e NCCL_DEBUG_SUBSYS=INIT,NET,ENV \
  -e GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
  -e NRL_REFIT_BUFFER_MEMORY_RATIO=0.05 \
  -e PYTHONUNBUFFERED=1 \
  -e HF_HOME=/work/hf \
  -e UV_CACHE_DIR=/work/uv-cache \
  -e VLLM_CACHE_ROOT=/work/vllm-cache \
  -e NEMO_RL_NVML_MEM_GET_INFO_FALLBACK=1 \
  "${IMAGE}" \
  bash -lc "ray stop --force >/dev/null 2>&1 || true; ray start --address=${HEAD_IP}:6379 --node-ip-address=${WORKER_IP} --dashboard-agent-listen-port=52365 --dashboard-agent-grpc-port=53007 --runtime-env-agent-port=53005 --node-manager-port=53001 --object-manager-port=53003 --metrics-export-port=53009 --min-worker-port=54001 --max-worker-port=54257 --num-gpus=1 --num-cpus=16 --object-store-memory=${RAY_OBJECT_STORE_MEMORY_BYTES} --disable-usage-stats --block" \
  > "${RUN_DIR}/worker.container.id"
EOF

  sleep 12
  ssh "${WORKER_SSH}" "docker logs '${WORKER_CONTAINER}' --tail 80" \
    | tee "${RUN_DIR}/logs/worker-start.log"
}

echo "starting Ray head container on ${HEAD_IP}"
docker run -d --name "${HEAD_CONTAINER}" \
  "${COMMON_DOCKER[@]}" \
  -v "${RUN_DIR}:/work" \
  "${COMMON_ENV[@]}" \
  "${IMAGE}" \
  bash -lc "ray stop --force >/dev/null 2>&1 || true; ray start --head --node-ip-address=${HEAD_IP} --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --dashboard-agent-listen-port=52365 --dashboard-agent-grpc-port=53007 --runtime-env-agent-port=53005 --node-manager-port=53001 --object-manager-port=53003 --metrics-export-port=53009 --min-worker-port=54001 --max-worker-port=54257 --num-gpus=1 --num-cpus=16 --object-store-memory=${RAY_OBJECT_STORE_MEMORY_BYTES} --disable-usage-stats --block" \
  > "${RUN_DIR}/head.container.id"

sleep 8
docker logs "${HEAD_CONTAINER}" --tail 80 | tee "${RUN_DIR}/logs/head-start.log"

start_worker_container

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

echo "verifying Ray worker NCCL environment"
docker exec "${HEAD_CONTAINER}" bash -lc "RAY_ADDRESS=${HEAD_IP}:6379 python - <<'PY'
import json
import os

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

keys = [
    'NCCL_IB_DISABLE',
    'NCCL_SOCKET_IFNAME',
    'NCCL_SOCKET_FAMILY',
    'NCCL_NET',
    'NCCL_NET_PLUGIN',
    'NCCL_DEBUG',
    'NCCL_DEBUG_SUBSYS',
    'GLOO_SOCKET_IFNAME',
]

ray.init(address='auto')

@ray.remote(num_cpus=0)
def env_probe():
    return {
        'node_ip': ray._private.services.get_node_ip_address(),
        'env': {key: os.environ.get(key) for key in keys},
    }

results = []
for node in [node for node in ray.nodes() if node.get('Alive')]:
    results.append(
        ray.get(
            env_probe.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node['NodeID'], soft=False
                )
            ).remote()
        )
    )
print(json.dumps(results, sort_keys=True))
expected = {
    'NCCL_IB_DISABLE': '1',
    'NCCL_SOCKET_IFNAME': '${NCCL_SOCKET_IFNAME}',
    'NCCL_SOCKET_FAMILY': 'AF_INET',
    'NCCL_NET': 'Socket',
    'NCCL_NET_PLUGIN': 'none',
}
for result in results:
    for key, value in expected.items():
        if result['env'].get(key) != value:
            raise SystemExit(f'missing {key}={value} on {result}')
ray.shutdown()
PY" | tee "${RUN_DIR}/logs/ray-env.log"

echo "running NeMo native Async GRPO CLI smoke"
set +e
docker exec \
  -e RAY_ADDRESS="${HEAD_IP}:6379" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
  -e NCCL_SOCKET_FAMILY=AF_INET \
  -e NCCL_NET=Socket \
  -e NCCL_NET_PLUGIN=none \
  -e NCCL_DEBUG=INFO \
  -e NCCL_DEBUG_SUBSYS=INIT,NET,ENV \
  -e GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
  -e NRL_REFIT_BUFFER_MEMORY_RATIO=0.05 \
  -e PYTHONUNBUFFERED=1 \
  -e NEMO_RL_NVML_MEM_GET_INFO_FALLBACK=1 \
  "${HEAD_CONTAINER}" bash -lc "
    cd /opt/nemo-rl
    python examples/run_grpo.py \
      --config examples/configs/recipes/llm/grpo-qwen3-0.6b-1n8g-sglang.yaml \
      grpo.async_grpo.enabled=true \
      grpo.async_grpo.max_trajectory_age_steps=1 \
      grpo.async_grpo.in_flight_weight_updates=true \
      grpo.async_grpo.recompute_kv_cache_after_weight_updates=false \
      grpo.num_prompts_per_step=1 \
      grpo.num_generations_per_prompt=4 \
      grpo.max_num_steps=1 \
      grpo.max_num_epochs=1 \
      grpo.val_at_start=false \
      grpo.val_at_end=false \
      grpo.val_period=9999 \
      grpo.use_dynamic_sampling=false \
      grpo.reward_scaling.enabled=false \
      grpo.reward_shaping.enabled=false \
      loss_fn.reference_policy_kl_penalty=0.0 \
      loss_fn.use_importance_sampling_correction=true \
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
      data_plane.enabled=false \
      logger.log_dir=/work/logs/grpo \
      logger.wandb_enabled=false \
      logger.tensorboard_enabled=false \
      logger.mlflow_enabled=false \
      logger.swanlab_enabled=false \
      logger.monitor_gpus=false \
      checkpointing.enabled=false \
      cluster.gpus_per_node=1 \
      cluster.num_nodes=2 \
      ${EXTRA_OVERRIDES[*]}
  " 2>&1 | tee "${RUN_DIR}/logs/grpo-nemo-async.log"
code=${PIPESTATUS[0]}
set -e
echo "docker_exec_exit_code=${code}" | tee "${RUN_DIR}/logs/exit-code.log"

echo "collecting final logs"
docker logs "${HEAD_CONTAINER}" --tail 160 > "${RUN_DIR}/logs/head-final.log" 2>&1 || true
ssh "${WORKER_SSH}" "docker logs '${WORKER_CONTAINER}' --tail 160" > "${RUN_DIR}/logs/worker-final.log" 2>&1 || true
docker exec "${HEAD_CONTAINER}" bash -lc "ray list actors --address=${HEAD_IP}:6379 --format=json" > "${RUN_DIR}/logs/ray-actors-final.json" 2>/dev/null || true

ok=1
for pattern in \
  "Running async GRPO training" \
  "Started continuous background trajectory collection" \
  "Buffer ready! Starting training loop" \
  "Synchronizing policy weights to trajectory collector"; do
  if ! grep -q "${pattern}" "${RUN_DIR}/logs/grpo-nemo-async.log"; then
    echo "missing_success_pattern=${pattern}" | tee -a "${RUN_DIR}/logs/exit-code.log"
    ok=0
  fi
done

if ! HEAD_IP="${HEAD_IP}" WORKER_IP="${WORKER_IP}" HEAD_HOSTNAME="${HEAD_HOSTNAME}" WORKER_HOSTNAME="${WORKER_HOSTNAME}" LOG_PATH="${RUN_DIR}/logs/grpo-nemo-async.log" python3 - <<'PY' | tee -a "${RUN_DIR}/logs/exit-code.log"; then
import os
import re
import sys

head_ip = os.environ["HEAD_IP"]
worker_ip = os.environ["WORKER_IP"]
head_hostname = os.environ["HEAD_HOSTNAME"]
worker_hostname = os.environ["WORKER_HOSTNAME"]
valid_ips = {head_ip, worker_ip}
text = open(os.environ["LOG_PATH"], encoding="utf-8", errors="replace").read()

policy_ips = set(re.findall(r"DTensorPolicyWorkerV2[^\n]*ip=([0-9.]+)", text))
vllm_ips = set(re.findall(r"VllmAsyncGenerationWorker[^\n]*ip=([0-9.]+)", text))

for ip in valid_ips:
    if re.search(rf"\|\s*{re.escape(ip)}\s*\|\s*\('worker-0',\)", text):
        policy_ips.add(ip)
    if re.search(rf"DTensorPolicyWorkerV2[^\n]*NET/Socket : Using .*:{re.escape(ip)}<", text):
        policy_ips.add(ip)
    if re.search(rf"DTensorPolicyWorkerV2[^\n]*{re.escape(head_hostname)}:", text) and ip == head_ip:
        policy_ips.add(ip)
    if re.search(rf"DTensorPolicyWorkerV2[^\n]*{re.escape(worker_hostname)}:", text) and ip == worker_ip:
        policy_ips.add(ip)
    if re.search(rf"VllmAsyncGenerationWorker[^\n]*NET/Socket : Using .*:{re.escape(ip)}<", text):
        vllm_ips.add(ip)
    if re.search(rf"VllmAsyncGenerationWorker[^\n]*{re.escape(head_hostname)}:", text) and ip == head_ip:
        vllm_ips.add(ip)
    if re.search(rf"VllmAsyncGenerationWorker[^\n]*{re.escape(worker_hostname)}:", text) and ip == worker_ip:
        vllm_ips.add(ip)

print(f"topology_policy_ips={','.join(sorted(policy_ips)) or 'missing'}")
print(f"topology_vllm_ips={','.join(sorted(vllm_ips)) or 'missing'}")

if not policy_ips:
    print("missing_topology_pattern=policy_worker_ip")
    sys.exit(1)
if not vllm_ips:
    print("missing_topology_pattern=vllm_worker_ip")
    sys.exit(1)
if not policy_ips <= valid_ips or not vllm_ips <= valid_ips:
    print("unexpected_topology_ip=outside_two_sparks")
    sys.exit(1)
if policy_ips & vllm_ips:
    print("missing_topology_pattern=non_colocated_policy_and_vllm")
    sys.exit(1)
PY
  ok=0
fi

if [[ "${code}" -eq 0 && "${ok}" -eq 1 ]]; then
  printf 'SUCCEEDED %s\n' "$(date -Is)" | tee "${RUN_DIR}/status"
else
  printf 'FAILED code=%s %s\n' "${code}" "$(date -Is)" | tee "${RUN_DIR}/status"
fi

if [[ "${KEEP_CONTAINERS:-0}" != "1" ]]; then
  docker rm -f "${HEAD_CONTAINER}" >/dev/null 2>&1 || true
  ssh "${WORKER_SSH}" "docker rm -f '${WORKER_CONTAINER}' >/dev/null 2>&1 || true"
fi

exit "${code}"

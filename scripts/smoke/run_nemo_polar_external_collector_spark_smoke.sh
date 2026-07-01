#!/usr/bin/env bash
set -euo pipefail

# c236061b is the proven deployed NeMo RL revision (matches preflight_spark.sh's
# default). Older 37526dfa predates the use_cispo config key, so any cispo-axis
# spec relying on the default image fails with "Key 'use_cispo' is not in struct".
NEMO_RL_REF="${NEMO_RL_REF:-c236061b250e97638722292ab8a54d5eb47ae00f}"
IMAGE="${IMAGE:-local/nemo-rl-main-cu132:${NEMO_RL_REF:0:8}}"
HEAD_IP="${HEAD_IP:-192.168.100.10}"
WORKER_IP="${WORKER_IP:-192.168.100.11}"
WORKER_SSH="${WORKER_SSH:-jarrodbarnes@192.168.100.11}"
HEAD_HOSTNAME="${HEAD_HOSTNAME:-spark-f7e2}"
WORKER_HOSTNAME="${WORKER_HOSTNAME:-spark-cfd0}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}"
# Weight-sync NCCL transport. Default forces plain TCP sockets (IB disabled);
# set NCCL_TRANSPORT=roce to route the weight-sync broadcast over RDMA instead.
# GID index 3 = RoCEv2 IPv4-mapped on device rocep1s0f1 (both hosts).
NCCL_TRANSPORT="${NCCL_TRANSPORT:-socket}"
NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f1}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
if [[ "${NCCL_TRANSPORT}" == "roce" ]]; then
  NCCL_IB_DISABLE_VALUE="0"
  NCCL_NET_VALUE="IB"
  NCCL_NET_PLUGIN_VALUE=""
else
  NCCL_IB_DISABLE_VALUE="1"
  NCCL_NET_VALUE="Socket"
  NCCL_NET_PLUGIN_VALUE="none"
fi
# Cap Ray's object store: its default ~30% uncapped reservation competes with
# model weights and optimizer offload for DGX Spark's unified ~121.69GB pool.
# Async GRPO moves weights via NCCL broadcast, not plasma, so a few GB suffices.
RAY_OBJECT_STORE_MEMORY_BYTES="${RAY_OBJECT_STORE_MEMORY_BYTES:-8589934592}"
# Sizes the refit weight-broadcast bucket buffer (fraction of device memory per
# bucket, doubled by NRL_REFIT_NUM_BUFFERS). Broadcast spikes true host RSS in a
# way Docker cgroup accounting misses on this unified-memory host; override lower
# (e.g. 0.02, upstream default) to shrink the transient spike.
NRL_REFIT_BUFFER_MEMORY_RATIO="${NRL_REFIT_BUFFER_MEMORY_RATIO:-0.05}"
MODEL_HOST="${MODEL_HOST:-/home/jarrodbarnes/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
MODEL_CONT="${MODEL_CONT:-/host-hf/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
MODEL_MOUNT_HOST="${MODEL_MOUNT_HOST:-}"
MODEL_MOUNT_CONT="${MODEL_MOUNT_CONT:-}"
REPO_HOST="${REPO_HOST:-/home/jarrodbarnes/ProRL-Agent-Server}"
STAMP="${1:-$(date +%Y%m%d-%H%M%S)}"
shift || true
EXTRA_OVERRIDES=("$@")
RUN_DIR="${RUN_DIR:-/home/jarrodbarnes/nemo-rl-smoke/nemo-polar-qwen3-0p6b-${STAMP}}"
RUN_START_EPOCH="$(date +%s)"
RUN_START_TIME="$(date -Is)"
HEAD_CONTAINER="nemo-polar-head-${STAMP}"
WORKER_CONTAINER="nemo-polar-worker-${STAMP}"
POLAR_ROLLOUT_PORT="${POLAR_ROLLOUT_PORT:-19080}"
POLAR_GATEWAY_PORT="${POLAR_GATEWAY_PORT:-19100}"
VLLM_HTTP_PORT="${VLLM_HTTP_PORT:-31000}"
POLAR_DATASET_ID="${POLAR_DATASET_ID:-${TASK_DATASET_ID:-nvidia/Nemotron-RL-ReasoningGym-v1}}"
POLAR_DATASET_CONFIG="${POLAR_DATASET_CONFIG:-${TASK_DATASET_CONFIG:-default}}"
POLAR_DATASET_SPLIT="${POLAR_DATASET_SPLIT:-${TASK_DATASET_SPLIT:-train}}"
POLAR_DATASET_LIMIT="${POLAR_DATASET_LIMIT:-${TASK_DATASET_LIMIT:-2}}"
POLAR_DATASET_SCAN_ROWS="${POLAR_DATASET_SCAN_ROWS:-${TASK_DATASET_SCAN_ROWS:-200}}"
POLAR_SOURCE_DATASETS="${POLAR_SOURCE_DATASETS:-${POLAR_DATASET_SOURCE_DATASETS:-}}"
POLAR_DATASET_LOCAL_JSONL="${POLAR_DATASET_LOCAL_JSONL:-}"
POLAR_DATASET_ANSWER_FORMAT="${POLAR_DATASET_ANSWER_FORMAT:-none}"
POLAR_DATASET_RUNTIME_IMAGE="${POLAR_DATASET_RUNTIME_IMAGE:-${TASK_RUNTIME_IMAGE:-polar-spark-calculator:latest}}"
POLAR_MATRIX_NAME="${POLAR_MATRIX_NAME:-smoke}"
POLAR_MATRIX_CELL="${POLAR_MATRIX_CELL:-}"
POLAR_DATASET_FAMILY="${POLAR_DATASET_FAMILY:-nemo_gym}"
POLAR_VERIFIER_TYPE="${POLAR_VERIFIER_TYPE:-exact_answer}"
POLAR_EXECUTION_TYPE="${POLAR_EXECUTION_TYPE:-single_turn_chat}"
POLAR_DIFFICULTY_BAND="${POLAR_DIFFICULTY_BAND:-unspecified}"
POLAR_ADAPTER_NAME="${POLAR_ADAPTER_NAME:-nemo_gym_jsonl}"
POLAR_MATRIX_TAGS="${POLAR_MATRIX_TAGS:-}"
POLAR_MATRIX_NOTES="${POLAR_MATRIX_NOTES:-}"
POLAR_MODEL_NAME="${POLAR_MODEL_NAME:-Qwen/Qwen3-0.6B}"
POLAR_MODEL_MAX_NEW_TOKENS="${POLAR_MODEL_MAX_NEW_TOKENS:-1024}"
POLAR_MODEL_MAX_TOTAL_SEQUENCE_LENGTH="${POLAR_MODEL_MAX_TOTAL_SEQUENCE_LENGTH:-8192}"
POLAR_MODEL_MAX_MODEL_LEN="${POLAR_MODEL_MAX_MODEL_LEN:-8192}"
POLAR_MODEL_TEMPERATURE="${POLAR_MODEL_TEMPERATURE:-0.6}"
# top_p=1.0 keeps recomputed-logprob support identical to the sampler; nucleus truncation otherwise yields sparse -inf positions.
POLAR_MODEL_TOP_P="${POLAR_MODEL_TOP_P:-1.0}"
NEMO_VLLM_GPU_MEMORY_UTILIZATION="${NEMO_VLLM_GPU_MEMORY_UTILIZATION:-0.35}"
NEMO_VLLM_ENFORCE_EAGER="${NEMO_VLLM_ENFORCE_EAGER:-true}"
NEMO_VLLM_PRECISION="${NEMO_VLLM_PRECISION:-bfloat16}"
NEMO_VLLM_KV_CACHE_DTYPE="${NEMO_VLLM_KV_CACHE_DTYPE:-auto}"
NEMO_VLLM_MAX_NUM_SEQS="${NEMO_VLLM_MAX_NUM_SEQS:-}"
NEMO_VLLM_MAX_NUM_BATCHED_TOKENS="${NEMO_VLLM_MAX_NUM_BATCHED_TOKENS:-}"
NEMO_GRPO_NUM_PROMPTS_PER_STEP="${NEMO_GRPO_NUM_PROMPTS_PER_STEP:-1}"
NEMO_GRPO_NUM_GENERATIONS_PER_PROMPT="${NEMO_GRPO_NUM_GENERATIONS_PER_PROMPT:-2}"
NEMO_GRPO_MAX_NUM_STEPS="${NEMO_GRPO_MAX_NUM_STEPS:-1}"
NEMO_GRPO_MAX_TRAJECTORY_AGE_STEPS="${NEMO_GRPO_MAX_TRAJECTORY_AGE_STEPS:-1}"
NEMO_POLICY_TRAIN_GLOBAL_BATCH_SIZE="${NEMO_POLICY_TRAIN_GLOBAL_BATCH_SIZE:-$((NEMO_GRPO_NUM_PROMPTS_PER_STEP * NEMO_GRPO_NUM_GENERATIONS_PER_PROMPT))}"
NEMO_DATA_TRAIN_SPLIT_VALIDATION_SIZE="${NEMO_DATA_TRAIN_SPLIT_VALIDATION_SIZE:-0.5}"
NEMO_POLAR_GROUP_WORKERS="${NEMO_POLAR_GROUP_WORKERS:-${NEMO_GRPO_NUM_GENERATIONS_PER_PROMPT}}"
POLAR_GATEWAY_MAX_INIT_WORKERS="${POLAR_GATEWAY_MAX_INIT_WORKERS:-${NEMO_POLAR_GROUP_WORKERS}}"
POLAR_GATEWAY_MAX_RUN_WORKERS="${POLAR_GATEWAY_MAX_RUN_WORKERS:-${NEMO_POLAR_GROUP_WORKERS}}"
POLAR_GATEWAY_MAX_POSTRUN_WORKERS="${POLAR_GATEWAY_MAX_POSTRUN_WORKERS:-${NEMO_POLAR_GROUP_WORKERS}}"
POLAR_MODEL_REQUEST_TIMEOUT_SECONDS="${POLAR_MODEL_REQUEST_TIMEOUT_SECONDS:-180}"
POLAR_TASK_TIMEOUT_SECONDS="${POLAR_TASK_TIMEOUT_SECONDS:-300}"
CLEANUP_DONE=0

cleanup_smoke_resources() {
  local original_status=$?
  local cleanup_status=0
  local pids

  if [[ "${CLEANUP_DONE}" -eq 1 ]]; then
    return "${original_status}"
  fi
  CLEANUP_DONE=1

  set +e
  mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/polar"
  {
    echo "cleanup_start_time=$(date -Is)"

    for pid_file in "${RUN_DIR}/polar/rollout.pid" "${RUN_DIR}/polar/gateway.pid"; do
      if [[ -s "${pid_file}" ]]; then
        pid="$(cat "${pid_file}")"
        if [[ -n "${pid}" ]]; then
          kill "${pid}" >/dev/null 2>&1 || true
          sleep 1
          kill -0 "${pid}" >/dev/null 2>&1 && kill -9 "${pid}" >/dev/null 2>&1 || true
        fi
      fi
    done

    for port in "${POLAR_ROLLOUT_PORT}" "${POLAR_GATEWAY_PORT}"; do
      pids=$(lsof -ti "tcp:${port}" 2>/dev/null || true)
      if [[ -n "${pids}" ]]; then
        kill ${pids} >/dev/null 2>&1 || true
        sleep 1
        pids=$(lsof -ti "tcp:${port}" 2>/dev/null || true)
        if [[ -n "${pids}" ]]; then
          kill -9 ${pids} >/dev/null 2>&1 || true
        fi
      fi
    done

    docker rm -f "${HEAD_CONTAINER}" >/dev/null 2>&1 || true
    ssh "${WORKER_SSH}" "docker rm -f '${WORKER_CONTAINER}' >/dev/null 2>&1 || true"
    ssh "${WORKER_SSH}" "docker rm -f vllm-polar-qwen06 >/dev/null 2>&1 || true"

    if docker ps -a --format '{{.Names}}' | grep -Fx "${HEAD_CONTAINER}" >/dev/null; then
      echo "cleanup_leftover_container=${HEAD_CONTAINER}"
      cleanup_status=1
    fi
    if ssh "${WORKER_SSH}" "docker ps -a --format '{{.Names}}' | grep -Fx '${WORKER_CONTAINER}' >/dev/null"; then
      echo "cleanup_leftover_container=${WORKER_CONTAINER}"
      cleanup_status=1
    fi
    for port in "${POLAR_ROLLOUT_PORT}" "${POLAR_GATEWAY_PORT}"; do
      if lsof -ti "tcp:${port}" >/dev/null 2>&1; then
        echo "cleanup_leftover_port=${port}"
        cleanup_status=1
      fi
    done

    echo "cleanup_exit_code=${cleanup_status}"
    echo "cleanup_end_time=$(date -Is)"
  } >> "${RUN_DIR}/logs/cleanup.log" 2>&1

  if [[ "${original_status}" -ne 0 ]]; then
    return "${original_status}"
  fi
  return "${cleanup_status}"
}

echo "This training run is worth doing because it will improve the go/no-go decision for Jarrod's two-Spark small-model agentic RL lab as measured by live NeMo Async GRPO consuming non-forced Polar rollouts from a real agentic dataset with valid tokens, logprobs, masks, grouped rewards, replay-buffer sampling, and weight sync, producing a run manifest and next-ablation decision."
echo "run_dir=${RUN_DIR}"
echo "image=${IMAGE}"
echo "run_start_time=${RUN_START_TIME}"

if [[ -z "${MODEL_MOUNT_HOST}" || -z "${MODEL_MOUNT_CONT}" ]]; then
  if [[ "${MODEL_HOST}" == */hub/* ]]; then
    MODEL_MOUNT_HOST="${MODEL_HOST%/hub/*}/hub"
    MODEL_MOUNT_CONT="/host-hf/hub"
  else
    MODEL_MOUNT_HOST="${MODEL_HOST}"
    MODEL_MOUNT_CONT="${MODEL_CONT}"
  fi
fi

mkdir -p "${RUN_DIR}"/{data,logs,polar,ray-head,tmp,hf}
ssh "${WORKER_SSH}" "mkdir -p '${RUN_DIR}'/{data,logs,ray-worker,tmp,hf}"
trap cleanup_smoke_resources EXIT

rsync -az --delete \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude "web/node_modules" \
  --exclude "rollout_results" \
  --exclude "tmp" \
  "${REPO_HOST}/" "${WORKER_SSH}:${REPO_HOST}/"

cat > "${RUN_DIR}/nccl.conf" <<EOF
NCCL_IB_DISABLE=${NCCL_IB_DISABLE_VALUE}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}
NCCL_SOCKET_FAMILY=AF_INET
NCCL_NET=${NCCL_NET_VALUE}
NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN_VALUE}
NCCL_IB_HCA=${NCCL_IB_HCA}
NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET,ENV
EOF
scp -q "${RUN_DIR}/nccl.conf" "${WORKER_SSH}:${RUN_DIR}/nccl.conf"

if [[ -n "${POLAR_DATASET_ID}" && "${POLAR_DATASET_ID}" != "none" ]]; then
  SOURCE_ARGS=()
  if [[ -n "${POLAR_SOURCE_DATASETS}" ]]; then
    IFS=',' read -ra SOURCE_DATASET_ITEMS <<< "${POLAR_SOURCE_DATASETS}"
    for source_dataset in "${SOURCE_DATASET_ITEMS[@]}"; do
      SOURCE_ARGS+=(--source-dataset "${source_dataset}")
    done
  fi
  LOCAL_JSONL_ARGS=()
  if [[ -n "${POLAR_DATASET_LOCAL_JSONL}" ]]; then
    LOCAL_JSONL_ARGS+=(--local-jsonl "${POLAR_DATASET_LOCAL_JSONL}")
  fi
  MATRIX_TAG_ARGS=()
  if [[ -n "${POLAR_MATRIX_TAGS}" ]]; then
    IFS=',' read -ra MATRIX_TAG_ITEMS <<< "${POLAR_MATRIX_TAGS}"
    for matrix_tag in "${MATRIX_TAG_ITEMS[@]}"; do
      MATRIX_TAG_ARGS+=(--matrix-tag "${matrix_tag}")
    done
  fi
  PREPARE_VLLM_ARGS=(
    --vllm-gpu-memory-utilization "${NEMO_VLLM_GPU_MEMORY_UTILIZATION}"
    --vllm-enforce-eager "${NEMO_VLLM_ENFORCE_EAGER}"
    --vllm-precision "${NEMO_VLLM_PRECISION}"
    --vllm-kv-cache-dtype "${NEMO_VLLM_KV_CACHE_DTYPE}"
  )
  if [[ -n "${NEMO_VLLM_MAX_NUM_SEQS}" ]]; then
    PREPARE_VLLM_ARGS+=(--vllm-max-num-seqs "${NEMO_VLLM_MAX_NUM_SEQS}")
  fi
  if [[ -n "${NEMO_VLLM_MAX_NUM_BATCHED_TOKENS}" ]]; then
    PREPARE_VLLM_ARGS+=(--vllm-max-num-batched-tokens "${NEMO_VLLM_MAX_NUM_BATCHED_TOKENS}")
  fi
  # Prefer the repo venv over bare python3: a non-interactive SSH shell resolves
  # python3 to the system interpreter, which lacks the repo's pip extras (e.g. reasoning-gym).
  PREPARE_DATASET_PYTHON="python3"
  if [[ -x "${REPO_HOST}/.venv/bin/python3" ]]; then
    PREPARE_DATASET_PYTHON="${REPO_HOST}/.venv/bin/python3"
  fi
  echo "prepare_dataset interpreter: ${PREPARE_DATASET_PYTHON}"
  PYTHONPATH="${REPO_HOST}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PREPARE_DATASET_PYTHON}" -m nemo_polar_bridge.datasets.prepare_dataset \
    --dataset-id "${POLAR_DATASET_ID}" \
    --config "${POLAR_DATASET_CONFIG}" \
    --split "${POLAR_DATASET_SPLIT}" \
    --limit "${POLAR_DATASET_LIMIT}" \
    --scan-rows "${POLAR_DATASET_SCAN_ROWS}" \
    "${SOURCE_ARGS[@]}" \
    "${LOCAL_JSONL_ARGS[@]}" \
    --answer-format "${POLAR_DATASET_ANSWER_FORMAT}" \
    --output-dir "${RUN_DIR}" \
    --repo-host "${REPO_HOST}" \
    --runtime-image "${POLAR_DATASET_RUNTIME_IMAGE}" \
    --model-name "${POLAR_MODEL_NAME}" \
    --model-path "${MODEL_CONT}" \
    --image "${IMAGE}" \
    --nemo-rl-ref "${NEMO_RL_REF}" \
    --head-ip "${HEAD_IP}" \
    --worker-ip "${WORKER_IP}" \
    --head-hostname "${HEAD_HOSTNAME}" \
    --worker-hostname "${WORKER_HOSTNAME}" \
    --polar-rollout-port "${POLAR_ROLLOUT_PORT}" \
    --polar-gateway-port "${POLAR_GATEWAY_PORT}" \
    --vllm-http-port "${VLLM_HTTP_PORT}" \
    --matrix-name "${POLAR_MATRIX_NAME}" \
    --matrix-cell "${POLAR_MATRIX_CELL}" \
    --dataset-family "${POLAR_DATASET_FAMILY}" \
    --verifier-type "${POLAR_VERIFIER_TYPE}" \
    --execution-type "${POLAR_EXECUTION_TYPE}" \
    --difficulty-band "${POLAR_DIFFICULTY_BAND}" \
    --adapter-name "${POLAR_ADAPTER_NAME}" \
    --matrix-notes "${POLAR_MATRIX_NOTES}" \
    "${MATRIX_TAG_ARGS[@]}" \
    --model-max-tokens "${POLAR_MODEL_MAX_NEW_TOKENS}" \
    --model-temperature "${POLAR_MODEL_TEMPERATURE}" \
    --model-top-p "${POLAR_MODEL_TOP_P}" \
    --model-request-timeout-seconds "${POLAR_MODEL_REQUEST_TIMEOUT_SECONDS}" \
    --task-timeout-seconds "${POLAR_TASK_TIMEOUT_SECONDS}" \
    "${PREPARE_VLLM_ARGS[@]}" \
    --script-path "scripts/smoke/run_nemo_polar_external_collector_spark_smoke.sh"
else
cat > "${RUN_DIR}/data/train.jsonl" <<'JSONL'
{"input":"Polar external collector smoke prompt.","output":"ok"}
{"input":"Polar external collector smoke prompt 2.","output":"ok"}
JSONL

cat > "${RUN_DIR}/polar/topology.yaml" <<YAML
rollout:
  host: 0.0.0.0
  port: ${POLAR_ROLLOUT_PORT}
  public_url: http://${HEAD_IP}:${POLAR_ROLLOUT_PORT}
  save_dir: ${RUN_DIR}/polar/rollout_results
  dispatch_poll_interval_seconds: 0.5
  callback_grace_seconds: 60.0

gateway:
  heartbeat_interval_seconds: 5
  nodes:
    - id: nemo-polar-gateway
      host: 0.0.0.0
      port: ${POLAR_GATEWAY_PORT}
      public_url: http://${HEAD_IP}:${POLAR_GATEWAY_PORT}
      max_init_workers: ${POLAR_GATEWAY_MAX_INIT_WORKERS}
      max_run_workers: ${POLAR_GATEWAY_MAX_RUN_WORKERS}
      max_postrun_workers: ${POLAR_GATEWAY_MAX_POSTRUN_WORKERS}
      model_served: ${MODEL_CONT}
      inference:
        engine: vllm
        base_url: http://${HEAD_IP}:${VLLM_HTTP_PORT}
YAML

cat > "${RUN_DIR}/polar/attempt_matrix.json" <<'JSON'
[
  {"name": "pass", "force_fail": "0"},
  {"name": "fail", "force_fail": "1"}
]
JSON

cat > "${RUN_DIR}/polar/task_template.json" <<JSON
{
  "task_id": "nemo-polar-calculator-{weight_version}-{target_weight_version}-{group_id}-{attempt_index}-{name}",
  "instruction": "Call the model once through the Polar gateway, then edit calculator.py according to this attempt's ablation setting.",
  "timeout_seconds": 240.0,
  "runtime": {
    "backend": "docker",
    "image": "polar-spark-calculator:latest",
    "prepare": [
      {"type": "exec", "command": "rm -rf /polar/session/workspace && mkdir -p /polar/session/workspace /polar/session/logs/agent && cd /polar/session/workspace && git init -q && git config user.email 'polar@test' && git config user.name 'Polar'"},
      {"type": "upload_file", "source": "${REPO_HOST}/examples/calculator/assets/test_calculator.py", "target": "/polar/session/workspace/test_calculator.py"},
      {"type": "upload_file", "source": "${REPO_HOST}/examples/calculator/assets/calculator.py", "target": "/polar/session/workspace/calculator.py"},
      {"type": "exec", "command": "cd /polar/session/workspace && git add -A && git commit -qm 'initial'"}
    ],
    "network": "host",
    "workdir": "/polar/session/workspace"
  },
  "agent": {
    "harness": "shell",
    "model_name": "Qwen/Qwen3-0.6B",
    "custom_shell": {
      "command": "python3 - <<'PY'\nimport json\nimport os\nimport urllib.request\nbase = os.environ['OPENAI_BASE_URL'].rstrip('/')\napi_key = os.environ['OPENAI_API_KEY']\npayload = {'model': 'Qwen/Qwen3-0.6B', 'messages': [{'role': 'user', 'content': 'Return exactly: calculator smoke'}], 'max_tokens': 8, 'temperature': 1.0, 'top_p': 1.0}\nreq = urllib.request.Request(base + '/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + api_key}, method='POST')\nwith urllib.request.urlopen(req, timeout=120) as resp:\n    data = json.loads(resp.read())\nos.makedirs(os.environ.get('ARTIFACTS_DIR', '/polar/session/artifacts'), exist_ok=True)\nwith open(os.path.join(os.environ.get('ARTIFACTS_DIR', '/polar/session/artifacts'), 'llm_probe.json'), 'w') as fh:\n    json.dump(data, fh)\nif os.environ.get('POLAR_FORCE_FAIL') == '1':\n    raise SystemExit(0)\nsource = '''class Calculator:\n    def __call__(self, expression: str) -> int:\n        self.tokens = self._tokenize(expression)\n        self.index = 0\n        value = self._parse_expr()\n        if self.index != len(self.tokens):\n            raise ValueError(\"trailing input\")\n        return value\n\n    def _tokenize(self, expression: str) -> list[object]:\n        tokens: list[object] = []\n        i = 0\n        while i < len(expression):\n            char = expression[i]\n            if char.isspace():\n                i += 1\n                continue\n            if char.isdigit():\n                start = i\n                while i < len(expression) and expression[i].isdigit():\n                    i += 1\n                tokens.append(int(expression[start:i]))\n                continue\n            if char in \"+-*/()\":\n                tokens.append(char)\n                i += 1\n                continue\n            raise ValueError(f\"unexpected character: {char!r}\")\n        return tokens\n\n    def _peek(self) -> object | None:\n        if self.index >= len(self.tokens):\n            return None\n        return self.tokens[self.index]\n\n    def _consume(self, expected: object | None = None) -> object:\n        token = self._peek()\n        if token is None:\n            raise ValueError(\"unexpected end of input\")\n        if expected is not None and token != expected:\n            raise ValueError(f\"expected {expected!r}, got {token!r}\")\n        self.index += 1\n        return token\n\n    def _parse_expr(self) -> int:\n        value = self._parse_term()\n        while self._peek() in (\"+\", \"-\"):\n            op = self._consume()\n            rhs = self._parse_term()\n            if op == \"+\":\n                value += rhs\n            else:\n                value -= rhs\n        return value\n\n    def _parse_term(self) -> int:\n        value = self._parse_factor()\n        while self._peek() in (\"*\", \"/\"):\n            op = self._consume()\n            rhs = self._parse_factor()\n            if op == \"*\":\n                value *= rhs\n            else:\n                value //= rhs\n        return value\n\n    def _parse_factor(self) -> int:\n        token = self._peek()\n        if isinstance(token, int):\n            return self._consume()\n        if token == \"(\":\n            self._consume(\"(\")\n            value = self._parse_expr()\n            self._consume(\")\")\n            return value\n        raise ValueError(f\"unexpected token: {token!r}\")\n'''\nwith open('calculator.py', 'w') as fh:\n    fh.write(source)\nPY\npython3 test_calculator.py || true"
    }
  },
  "builder": {"strategy": "prefix_merging"},
  "evaluator": {
    "strategy": "test_on_output",
    "config": {
      "repo_dir": "/polar/session/workspace",
      "patch_command": "cd /polar/session/workspace && git add -A && git diff --cached --binary",
      "test_command": "cd /polar/session/workspace && python3 test_calculator.py && echo 'PASSED test_calculator'",
      "test_timeout": 60.0,
      "expected_output_json": {"test_calculator": "PASSED"},
      "exclude_patterns": ["__pycache__/**", "**/__pycache__/**", ".pytest_cache/**", "**/.pytest_cache/**"]
    },
    "refresh_runtime": true
  },
  "metadata": {
    "purpose": "NeMo native Async GRPO with external Polar collector smoke"
  }
}
JSON
fi
if [[ ! -f "${RUN_DIR}/polar/topology.yaml" ]]; then
cat > "${RUN_DIR}/polar/topology.yaml" <<YAML
rollout:
  host: 0.0.0.0
  port: ${POLAR_ROLLOUT_PORT}
  public_url: http://${HEAD_IP}:${POLAR_ROLLOUT_PORT}
  save_dir: ${RUN_DIR}/polar/rollout_results
  dispatch_poll_interval_seconds: 0.5
  callback_grace_seconds: 60.0

gateway:
  heartbeat_interval_seconds: 5
  nodes:
    - id: nemo-polar-gateway
      host: 0.0.0.0
      port: ${POLAR_GATEWAY_PORT}
      public_url: http://${HEAD_IP}:${POLAR_GATEWAY_PORT}
      max_init_workers: ${POLAR_GATEWAY_MAX_INIT_WORKERS}
      max_run_workers: ${POLAR_GATEWAY_MAX_RUN_WORKERS}
      max_postrun_workers: ${POLAR_GATEWAY_MAX_POSTRUN_WORKERS}
      model_served: ${MODEL_CONT}
      inference:
        engine: vllm
        base_url: http://${HEAD_IP}:${VLLM_HTTP_PORT}
YAML
fi
scp -q "${RUN_DIR}/data/train.jsonl" "${WORKER_SSH}:${RUN_DIR}/data/train.jsonl"
rsync -az "${RUN_DIR}/polar/" "${WORKER_SSH}:${RUN_DIR}/polar/"

docker rm -f "${HEAD_CONTAINER}" >/dev/null 2>&1 || true
ssh "${WORKER_SSH}" "docker rm -f '${WORKER_CONTAINER}' >/dev/null 2>&1 || true"
ssh "${WORKER_SSH}" "docker rm -f vllm-polar-qwen06 >/dev/null 2>&1 || true"

for port in "${POLAR_ROLLOUT_PORT}" "${POLAR_GATEWAY_PORT}"; do
  pids=$(lsof -ti "tcp:${port}" 2>/dev/null || true)
  if [[ -n "${pids}" ]]; then
    kill ${pids} || true
    sleep 1
  fi
done

cd "${REPO_HOST}"
nohup .venv/bin/polar serve_rollout -c "${RUN_DIR}/polar/topology.yaml" \
  > "${RUN_DIR}/logs/polar-rollout.log" 2>&1 &
echo $! > "${RUN_DIR}/polar/rollout.pid"
nohup .venv/bin/polar serve_gateway -c "${RUN_DIR}/polar/topology.yaml" --node-id nemo-polar-gateway \
  > "${RUN_DIR}/logs/polar-gateway.log" 2>&1 &
echo $! > "${RUN_DIR}/polar/gateway.pid"

for _ in $(seq 1 60); do
  if curl -sf "http://${HEAD_IP}:${POLAR_ROLLOUT_PORT}/health" >/dev/null; then
    break
  fi
  sleep 1
done
curl -sf "http://${HEAD_IP}:${POLAR_ROLLOUT_PORT}/nodes" | tee "${RUN_DIR}/logs/polar-nodes-start.json" || true

COMMON_ENV=(
  -e NVIDIA_VISIBLE_DEVICES=all
  -e CUDA_VISIBLE_DEVICES=0
  -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE_VALUE}
  -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}
  -e NCCL_SOCKET_FAMILY=AF_INET
  -e NCCL_NET=${NCCL_NET_VALUE}
  -e NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN_VALUE}
  -e NCCL_IB_HCA=${NCCL_IB_HCA}
  -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}
  -e NCCL_DEBUG=INFO
  -e NCCL_DEBUG_SUBSYS=INIT,NET,ENV
  -e GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}
  -e NRL_REFIT_BUFFER_MEMORY_RATIO=${NRL_REFIT_BUFFER_MEMORY_RATIO}
  -e PYTHONUNBUFFERED=1
  -e HF_HOME=/work/hf
  -e UV_CACHE_DIR=/work/uv-cache
  -e VLLM_CACHE_ROOT=/work/vllm-cache
  -e NEMO_RL_NVML_MEM_GET_INFO_FALLBACK=1
  -e PYTHONPATH=/work/ProRL-Agent-Server/src:/opt/nemo-rl
  -e NEMO_POLAR_ROLLOUT_URL=http://${HEAD_IP}:${POLAR_ROLLOUT_PORT}
  -e NEMO_POLAR_GATEWAY_URL=http://${HEAD_IP}:${POLAR_GATEWAY_PORT}
  -e NEMO_POLAR_TASK_TEMPLATE_PATH=/work/polar/task_template.json
  -e NEMO_POLAR_ATTEMPT_MATRIX_PATH=/work/polar/attempt_matrix.json
  -e NEMO_POLAR_COLLECTOR_NODE_IP=${HEAD_IP}
  -e NEMO_POLAR_TRAIN_NODE_IP=${WORKER_IP}
  -e NEMO_POLAR_INFERENCE_NODE_IP=${HEAD_IP}
  -e NEMO_POLAR_GROUP_WORKERS=${NEMO_POLAR_GROUP_WORKERS}
)

COMMON_DOCKER=(
  --gpus all
  --network host
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  --shm-size=32g
  --cap-add=IPC_LOCK
  # Pass each IB device via its own --device flag: a bind-mount of /dev/infiniband
  # makes the nodes visible but doesn't update Docker's device cgroup whitelist,
  # so libibverbs open()/ioctl() fails ("Failed to open device").
  --device=/dev/infiniband/uverbs0
  --device=/dev/infiniband/uverbs1
  --device=/dev/infiniband/uverbs2
  --device=/dev/infiniband/uverbs3
  --device=/dev/infiniband/rdma_cm
  -v "${MODEL_MOUNT_HOST}:${MODEL_MOUNT_CONT}:ro"
  -v "${RUN_DIR}/nccl.conf:/etc/nccl.conf:ro"
  -v "${REPO_HOST}:/work/ProRL-Agent-Server:ro"
  -v "${RUN_DIR}:/work"
)

echo "starting Ray head container on ${HEAD_IP}"
docker run -d --name "${HEAD_CONTAINER}" \
  "${COMMON_DOCKER[@]}" \
  "${COMMON_ENV[@]}" \
  "${IMAGE}" \
  bash -lc "ray stop --force >/dev/null 2>&1 || true; ray start --head --node-ip-address=${HEAD_IP} --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --dashboard-agent-listen-port=52365 --dashboard-agent-grpc-port=53007 --runtime-env-agent-port=53005 --node-manager-port=53001 --object-manager-port=53003 --metrics-export-port=53009 --min-worker-port=54001 --max-worker-port=54257 --num-gpus=1 --num-cpus=16 --object-store-memory=${RAY_OBJECT_STORE_MEMORY_BYTES} --disable-usage-stats --block" \
  > "${RUN_DIR}/head.container.id"

sleep 8
docker logs "${HEAD_CONTAINER}" --tail 80 | tee "${RUN_DIR}/logs/head-start.log"

echo "starting Ray worker container on ${WORKER_IP}"
ssh "${WORKER_SSH}" bash -s <<EOF
set -euo pipefail
docker run -d --name "${WORKER_CONTAINER}" \
  --gpus all --network host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=32g \
  --cap-add=IPC_LOCK \
  --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/uverbs1 \
  --device=/dev/infiniband/uverbs2 --device=/dev/infiniband/uverbs3 \
  --device=/dev/infiniband/rdma_cm \
  -v "${MODEL_MOUNT_HOST}:${MODEL_MOUNT_CONT}:ro" \
  -v "${RUN_DIR}:/work" \
  -v "${RUN_DIR}/nccl.conf:/etc/nccl.conf:ro" \
  -v "${REPO_HOST}:/work/ProRL-Agent-Server:ro" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE_VALUE} \
  -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
  -e NCCL_SOCKET_FAMILY=AF_INET \
  -e NCCL_NET=${NCCL_NET_VALUE} \
  -e NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN_VALUE} \
  -e NCCL_IB_HCA=${NCCL_IB_HCA} \
  -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} \
  -e NCCL_DEBUG=INFO \
  -e NCCL_DEBUG_SUBSYS=INIT,NET,ENV \
  -e GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
  -e NRL_REFIT_BUFFER_MEMORY_RATIO=${NRL_REFIT_BUFFER_MEMORY_RATIO} \
  -e PYTHONUNBUFFERED=1 \
  -e HF_HOME=/work/hf \
  -e UV_CACHE_DIR=/work/uv-cache \
  -e VLLM_CACHE_ROOT=/work/vllm-cache \
  -e NEMO_RL_NVML_MEM_GET_INFO_FALLBACK=1 \
  -e PYTHONPATH=/work/ProRL-Agent-Server/src:/opt/nemo-rl \
  -e NEMO_POLAR_ROLLOUT_URL=http://${HEAD_IP}:${POLAR_ROLLOUT_PORT} \
  -e NEMO_POLAR_GATEWAY_URL=http://${HEAD_IP}:${POLAR_GATEWAY_PORT} \
  -e NEMO_POLAR_TASK_TEMPLATE_PATH=/work/polar/task_template.json \
  -e NEMO_POLAR_ATTEMPT_MATRIX_PATH=/work/polar/attempt_matrix.json \
  -e NEMO_POLAR_COLLECTOR_NODE_IP=${HEAD_IP} \
  -e NEMO_POLAR_TRAIN_NODE_IP=${WORKER_IP} \
  -e NEMO_POLAR_INFERENCE_NODE_IP=${HEAD_IP} \
  -e NEMO_POLAR_GROUP_WORKERS=${NEMO_POLAR_GROUP_WORKERS} \
  "${IMAGE}" \
  bash -lc "ray stop --force >/dev/null 2>&1 || true; ray start --address=${HEAD_IP}:6379 --node-ip-address=${WORKER_IP} --dashboard-agent-listen-port=52365 --dashboard-agent-grpc-port=53007 --runtime-env-agent-port=53005 --node-manager-port=53001 --object-manager-port=53003 --metrics-export-port=53009 --min-worker-port=54001 --max-worker-port=54257 --num-gpus=1 --num-cpus=16 --object-store-memory=${RAY_OBJECT_STORE_MEMORY_BYTES} --disable-usage-stats --block" \
  > "${RUN_DIR}/worker.container.id"
EOF

sleep 12
ssh "${WORKER_SSH}" "docker logs '${WORKER_CONTAINER}' --tail 80" \
  | tee "${RUN_DIR}/logs/worker-start.log"

echo "verifying Ray cluster resources"
docker exec "${HEAD_CONTAINER}" bash -lc "RAY_ADDRESS=${HEAD_IP}:6379 python - <<'PY'
import json
import ray

ray.init(address='auto')
resources = ray.cluster_resources()
nodes = ray.nodes()
print(json.dumps(resources, sort_keys=True))
print(json.dumps(nodes, default=str, sort_keys=True))
if int(resources.get('GPU', 0)) < 2:
    raise SystemExit(f'expected >=2 GPUs, got {resources}')
if len([node for node in nodes if node.get('Alive')]) < 2:
    raise SystemExit('expected >=2 alive nodes')
ray.shutdown()
PY" | tee "${RUN_DIR}/logs/ray-resources.log"

VLLM_HYDRA_OVERRIDES=(
  "policy.generation.vllm_cfg.gpu_memory_utilization=${NEMO_VLLM_GPU_MEMORY_UTILIZATION}"
  "policy.generation.vllm_cfg.enforce_eager=${NEMO_VLLM_ENFORCE_EAGER}"
  "policy.generation.vllm_cfg.precision=${NEMO_VLLM_PRECISION}"
  "policy.generation.vllm_cfg.kv_cache_dtype=${NEMO_VLLM_KV_CACHE_DTYPE}"
)
if [[ -n "${NEMO_VLLM_MAX_NUM_SEQS}" ]]; then
  VLLM_HYDRA_OVERRIDES+=("++policy.generation.vllm_cfg.max_num_seqs=${NEMO_VLLM_MAX_NUM_SEQS}")
fi
if [[ -n "${NEMO_VLLM_MAX_NUM_BATCHED_TOKENS}" ]]; then
  VLLM_HYDRA_OVERRIDES+=(
    "++policy.generation.vllm_cfg.max_num_batched_tokens=${NEMO_VLLM_MAX_NUM_BATCHED_TOKENS}"
  )
fi

echo "running NeMo native Async GRPO with Polar external collector"
set +e
docker exec \
  -e RAY_ADDRESS="${HEAD_IP}:6379" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE_VALUE} \
  -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
  -e NCCL_SOCKET_FAMILY=AF_INET \
  -e NCCL_NET=${NCCL_NET_VALUE} \
  -e NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN_VALUE} \
  -e NCCL_IB_HCA=${NCCL_IB_HCA} \
  -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} \
  -e NCCL_DEBUG=INFO \
  -e NCCL_DEBUG_SUBSYS=INIT,NET,ENV \
  -e GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
  -e NRL_REFIT_BUFFER_MEMORY_RATIO=${NRL_REFIT_BUFFER_MEMORY_RATIO} \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/work/ProRL-Agent-Server/src:/opt/nemo-rl \
  -e NEMO_POLAR_ROLLOUT_URL=http://${HEAD_IP}:${POLAR_ROLLOUT_PORT} \
  -e NEMO_POLAR_GATEWAY_URL=http://${HEAD_IP}:${POLAR_GATEWAY_PORT} \
  -e NEMO_POLAR_TASK_TEMPLATE_PATH=/work/polar/task_template.json \
  -e NEMO_POLAR_ATTEMPT_MATRIX_PATH=/work/polar/attempt_matrix.json \
  -e NEMO_POLAR_COLLECTOR_NODE_IP=${HEAD_IP} \
  -e NEMO_POLAR_TRAIN_NODE_IP=${WORKER_IP} \
  -e NEMO_POLAR_INFERENCE_NODE_IP=${HEAD_IP} \
  -e NEMO_POLAR_GROUP_WORKERS=${NEMO_POLAR_GROUP_WORKERS} \
  "${HEAD_CONTAINER}" bash -lc "
    cd /opt/nemo-rl
    python -m nemo_polar_bridge.run_grpo_with_polar \
      --config examples/configs/recipes/llm/grpo-qwen3-0.6b-1n8g-sglang.yaml \
      grpo.async_grpo.enabled=true \
      grpo.async_grpo.max_trajectory_age_steps=${NEMO_GRPO_MAX_TRAJECTORY_AGE_STEPS} \
      grpo.async_grpo.in_flight_weight_updates=true \
      grpo.async_grpo.recompute_kv_cache_after_weight_updates=false \
      grpo.num_prompts_per_step=${NEMO_GRPO_NUM_PROMPTS_PER_STEP} \
      grpo.num_generations_per_prompt=${NEMO_GRPO_NUM_GENERATIONS_PER_PROMPT} \
      grpo.max_num_steps=${NEMO_GRPO_MAX_NUM_STEPS} \
      grpo.max_num_epochs=1 \
      grpo.val_at_start=false \
      grpo.val_at_end=false \
      grpo.val_period=9999 \
      grpo.overlong_filtering=false \
      grpo.use_dynamic_sampling=false \
      grpo.reward_scaling.enabled=false \
      grpo.reward_shaping.enabled=false \
      loss_fn.reference_policy_kl_penalty=0.0 \
      loss_fn.use_importance_sampling_correction=true \
      policy.model_name=${MODEL_CONT} \
      policy.tokenizer.name=${MODEL_CONT} \
      +policy.dtensor_cfg.automodel_kwargs.force_hf=true \
      policy.train_global_batch_size=${NEMO_POLICY_TRAIN_GLOBAL_BATCH_SIZE} \
      policy.train_micro_batch_size=1 \
      policy.logprob_batch_size=1 \
      policy.generation_batch_size=1 \
      policy.max_total_sequence_length=${POLAR_MODEL_MAX_TOTAL_SEQUENCE_LENGTH} \
      policy.sequence_packing.enabled=false \
      policy.dynamic_batching.enabled=false \
      policy.generation.backend=vllm \
      policy.generation.max_new_tokens=${POLAR_MODEL_MAX_NEW_TOKENS} \
      policy.generation.temperature=${POLAR_MODEL_TEMPERATURE} \
      policy.generation.top_p=${POLAR_MODEL_TOP_P} \
      policy.generation.top_k=null \
      policy.generation.port_range_low=${VLLM_HTTP_PORT} \
      policy.generation.port_range_high=$((VLLM_HTTP_PORT + 1)) \
      policy.generation.vllm_cfg.async_engine=true \
      +policy.generation.vllm_cfg.expose_http_server=true \
      policy.generation.vllm_cfg.enable_vllm_metrics_logger=false \
      policy.generation.vllm_cfg.max_model_len=${POLAR_MODEL_MAX_MODEL_LEN} \
      +policy.generation.vllm_kwargs.generation_config=vllm \
      policy.generation.colocated.enabled=false \
      policy.generation.colocated.resources.gpus_per_node=1 \
      policy.generation.colocated.resources.num_nodes=1 \
      data.train.dataset_name=ResponseDataset \
      +data.train.data_path=/work/data/train.jsonl \
      +data.train.input_key=input \
      +data.train.output_key=output \
      data.train.split_validation_size=${NEMO_DATA_TRAIN_SPLIT_VALIDATION_SIZE} \
      data.default.processor=math_hf_data_processor \
      data.default.env_name=math \
      env.math.num_workers=1 \
      data_plane.enabled=false \
      logger.log_dir=/work/logs/grpo-polar \
      logger.wandb_enabled=false \
      logger.tensorboard_enabled=false \
      logger.mlflow_enabled=false \
      logger.swanlab_enabled=false \
      logger.monitor_gpus=false \
      checkpointing.enabled=false \
      cluster.gpus_per_node=1 \
      cluster.num_nodes=2 \
      ${VLLM_HYDRA_OVERRIDES[*]} \
      ${EXTRA_OVERRIDES[*]}
  " 2>&1 | tee "${RUN_DIR}/logs/grpo-nemo-polar.log"
code=${PIPESTATUS[0]}
set -e
RUN_END_EPOCH="$(date +%s)"
RUN_END_TIME="$(date -Is)"
RUN_WALLCLOCK_SECONDS="$((RUN_END_EPOCH - RUN_START_EPOCH))"
echo "docker_exec_exit_code=${code}" | tee "${RUN_DIR}/logs/exit-code.log"
echo "run_end_time=${RUN_END_TIME}" | tee -a "${RUN_DIR}/logs/exit-code.log"
echo "run_wallclock_seconds=${RUN_WALLCLOCK_SECONDS}" | tee -a "${RUN_DIR}/logs/exit-code.log"

python3 - "${RUN_DIR}" "${RUN_START_TIME}" "${RUN_END_TIME}" "${RUN_WALLCLOCK_SECONDS}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
start_time = sys.argv[2]
end_time = sys.argv[3]
wallclock_seconds = int(sys.argv[4])
timing = {
    "event": "run_wallclock",
    "start_time": start_time,
    "end_time": end_time,
    "wallclock_seconds": wallclock_seconds,
}
(run_dir / "run_timing.json").write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n")
metrics_path = run_dir / "metrics.jsonl"
with metrics_path.open("a") as metrics_file:
    metrics_file.write(json.dumps(timing, sort_keys=True) + "\n")
config_path = run_dir / "config.json"
if config_path.exists():
    config = json.loads(config_path.read_text())
    config["timing"] = timing
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY

python3 "${REPO_HOST}/scripts/smoke/audit_nemo_polar_run.py" "${RUN_DIR}" || true

docker logs "${HEAD_CONTAINER}" --tail 160 > "${RUN_DIR}/logs/head-final.log" 2>&1 || true
ssh "${WORKER_SSH}" "docker logs '${WORKER_CONTAINER}' --tail 160" > "${RUN_DIR}/logs/worker-final.log" 2>&1 || true
curl -sf "http://${HEAD_IP}:${POLAR_ROLLOUT_PORT}/tasks?limit=20" \
  > "${RUN_DIR}/logs/polar-tasks-final.json" || true
curl -sf "http://${HEAD_IP}:${POLAR_GATEWAY_PORT}/admin/inference/status" \
  > "${RUN_DIR}/logs/polar-gateway-inference-final.json" || true

ok=1
for pattern in \
  "Running async GRPO training" \
  "Using Polar external rollout collector" \
  "Started Polar external trajectory collection" \
  "Polar collector adding group" \
  "ReplayBuffer.add: Adding trajectory" \
  "Sampled " \
  "Synchronizing policy weights to trajectory collector" \
  "Async GRPO training complete"; do
  if ! grep -q "${pattern}" "${RUN_DIR}/logs/grpo-nemo-polar.log"; then
    echo "missing_success_pattern=${pattern}" | tee -a "${RUN_DIR}/logs/exit-code.log"
    ok=0
  fi
done

if ! cleanup_smoke_resources; then
  echo "cleanup_failed=1" | tee -a "${RUN_DIR}/logs/exit-code.log"
  ok=0
fi

if [[ "${code}" -ne 0 || "${ok}" -ne 1 ]]; then
  echo "FAILED ${RUN_DIR}" | tee -a "${RUN_DIR}/logs/exit-code.log"
  exit 1
fi

echo "SUCCEEDED ${RUN_DIR}" | tee -a "${RUN_DIR}/logs/exit-code.log"

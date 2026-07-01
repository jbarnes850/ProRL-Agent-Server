#!/usr/bin/env bash
# Read-only infrastructure gate for the two-Spark NeMo Async GRPO + Polar external
# collector path: every check is a query, nothing is started/stopped/mutated. Run
# before any GPU spend on either host. Mirrors the *_SSH/*_IP/*_HOST env-var
# contract of the smoke scripts; override those vars for a different topology.
set -uo pipefail

NEMO_RL_REF="${NEMO_RL_REF:-c236061b250e97638722292ab8a54d5eb47ae00f}"
IMAGE="${IMAGE:-local/nemo-rl-main-cu132:${NEMO_RL_REF:0:8}}"
# HEAD_SSH/WORKER_SSH use Tailscale aliases: the worker's LAN-only IP is
# routable only from the head, not from this controller (direct SSH times out).
HEAD_SSH="${HEAD_SSH:-spark}"
WORKER_SSH="${WORKER_SSH:-spark-cfd0}"
HEAD_IP="${HEAD_IP:-192.168.100.10}"
WORKER_IP="${WORKER_IP:-192.168.100.11}"
HEAD_HOSTNAME="${HEAD_HOSTNAME:-spark-f7e2}"
WORKER_HOSTNAME="${WORKER_HOSTNAME:-spark-cfd0}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}"
REPO_HOST="${REPO_HOST:-/home/jarrodbarnes/ProRL-Agent-Server}"
MODEL_HOST="${MODEL_HOST:-/home/jarrodbarnes/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
POLAR_DATASET_RUNTIME_IMAGE="${POLAR_DATASET_RUNTIME_IMAGE:-polar-spark-calculator:latest}"
POLAR_ROLLOUT_PORT="${POLAR_ROLLOUT_PORT:-19080}"
POLAR_GATEWAY_PORT="${POLAR_GATEWAY_PORT:-19100}"
VLLM_HTTP_PORT="${VLLM_HTTP_PORT:-31000}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"

pass_count=0
fail_count=0
warn_count=0

check() {
  # check <label> <command...>  -- runs on the controller (no ssh wrapping)
  local label="$1"
  shift
  if "$@" >/tmp/preflight_check.out 2>&1; then
    echo "PASS  ${label}"
    pass_count=$((pass_count + 1))
  else
    echo "FAIL  ${label}"
    sed 's/^/      /' /tmp/preflight_check.out
    fail_count=$((fail_count + 1))
  fi
}

check_remote() {
  # check_remote <label> <ssh-target> <remote-command>
  local label="$1" target="$2" remote_cmd="$3"
  if ssh -o BatchMode=yes -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}" "${target}" "${remote_cmd}" \
      >/tmp/preflight_check.out 2>&1; then
    echo "PASS  ${label}"
    pass_count=$((pass_count + 1))
  else
    echo "FAIL  ${label}"
    sed 's/^/      /' /tmp/preflight_check.out
    fail_count=$((fail_count + 1))
  fi
}

warn_remote() {
  # warn_remote <label> <ssh-target> <remote-command> -- informational only, never fails the gate
  local label="$1" target="$2" remote_cmd="$3"
  if ssh -o BatchMode=yes -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}" "${target}" "${remote_cmd}" \
      >/tmp/preflight_check.out 2>&1; then
    echo "PASS  ${label}"
    pass_count=$((pass_count + 1))
  else
    echo "WARN  ${label}"
    sed 's/^/      /' /tmp/preflight_check.out
    warn_count=$((warn_count + 1))
  fi
}

echo "== two-Spark preflight =="
echo "head=${HEAD_SSH} (${HEAD_HOSTNAME}, ${HEAD_IP})  worker=${WORKER_SSH} (${WORKER_HOSTNAME}, ${WORKER_IP})"
echo "image=${IMAGE}  nemo_rl_ref=${NEMO_RL_REF}"
echo

echo "-- ssh reachability --"
check_remote "ssh reachable: head (${HEAD_SSH})" "${HEAD_SSH}" "true"
check_remote "ssh reachable: worker (${WORKER_SSH})" "${WORKER_SSH}" "true"

echo
echo "-- pinned NeMo RL image present --"
check_remote "image on head: ${IMAGE}" "${HEAD_SSH}" \
  "docker image inspect '${IMAGE}' >/dev/null"
check_remote "image on worker: ${IMAGE}" "${WORKER_SSH}" \
  "docker image inspect '${IMAGE}' >/dev/null"

echo
echo "-- Polar task-runtime image present (head only) --"
check_remote "image on head: ${POLAR_DATASET_RUNTIME_IMAGE}" "${HEAD_SSH}" \
  "docker image inspect '${POLAR_DATASET_RUNTIME_IMAGE}' >/dev/null"

echo
echo "-- model snapshot present --"
check_remote "model snapshot on head: ${MODEL_HOST}" "${HEAD_SSH}" \
  "test -d '${MODEL_HOST}'"
check_remote "model snapshot on worker: ${MODEL_HOST}" "${WORKER_SSH}" \
  "test -d '${MODEL_HOST}'"

echo
echo "-- repo checkout + polar CLI on head --"
check_remote "repo present: ${REPO_HOST}" "${HEAD_SSH}" \
  "test -d '${REPO_HOST}'"
check_remote "polar CLI executable: ${REPO_HOST}/.venv/bin/polar" "${HEAD_SSH}" \
  "test -x '${REPO_HOST}/.venv/bin/polar'"

echo
echo "-- RoCE / NCCL interface up --"
check_remote "ibv_devices lists an active-looking device (head)" "${HEAD_SSH}" \
  "ibv_devices | tail -n +3 | grep -q ."
check_remote "rdma: ${NCCL_SOCKET_IFNAME} ACTIVE (head)" "${HEAD_SSH}" \
  "rdma link show | grep -E \"netdev ${NCCL_SOCKET_IFNAME}([[:space:]]|\\\$)\" | grep -q 'state ACTIVE'"
check_remote "ibv_devices lists an active-looking device (worker)" "${WORKER_SSH}" \
  "ibv_devices | tail -n +3 | grep -q ."
check_remote "rdma: ${NCCL_SOCKET_IFNAME} ACTIVE (worker)" "${WORKER_SSH}" \
  "rdma link show | grep -E \"netdev ${NCCL_SOCKET_IFNAME}([[:space:]]|\\\$)\" | grep -q 'state ACTIVE'"

echo
echo "-- smoke ports free on head (${POLAR_ROLLOUT_PORT}, ${POLAR_GATEWAY_PORT}, ${VLLM_HTTP_PORT}) --"
for port in "${POLAR_ROLLOUT_PORT}" "${POLAR_GATEWAY_PORT}" "${VLLM_HTTP_PORT}"; do
  check_remote "port ${port} free on head" "${HEAD_SSH}" \
    "! (command -v ss >/dev/null && ss -ltn | awk '{print \$4}' | grep -q \":${port}\\\$\")"
done

echo
echo "-- no stale smoke containers or Ray processes (informational) --"
warn_remote "no nemo-polar-*/nemo-async-*/vllm-polar-* containers on head" "${HEAD_SSH}" \
  "! docker ps -a --format '{{.Names}}' | grep -qE '^(nemo-polar-|nemo-async-|vllm-polar-)'"
warn_remote "no nemo-polar-*/nemo-async-*/vllm-polar-* containers on worker" "${WORKER_SSH}" \
  "! docker ps -a --format '{{.Names}}' | grep -qE '^(nemo-polar-|nemo-async-|vllm-polar-)'"

echo
echo "== summary: ${pass_count} passed, ${fail_count} failed, ${warn_count} warned =="
rm -f /tmp/preflight_check.out

if [[ "${fail_count}" -gt 0 ]]; then
  echo "preflight FAILED -- do not launch a GPU run until every FAIL above is resolved."
  exit 1
fi
echo "preflight PASSED -- infra hard gates satisfied. Stale-resource WARNs above still deserve a manual look."
exit 0

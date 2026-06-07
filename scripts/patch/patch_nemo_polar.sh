#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
from __future__ import annotations

import importlib.util
from pathlib import Path


spec = importlib.util.find_spec("nemo_rl")
if spec is None or spec.submodule_search_locations is None:
    raise SystemExit("nemo_rl is not importable; run inside the NeMo RL environment")

root = Path(next(iter(spec.submodule_search_locations)))
grpo_sync = root / "algorithms" / "grpo_sync.py"
registry = root / "distributed" / "ray_actor_environment_registry.py"
nvml = root / "utils" / "nvml.py"
transfer_queue_adapter = root / "data_plane" / "adapters" / "transfer_queue.py"

required_tq_paths = [
    grpo_sync,
    root / "experience" / "sync_rollout_actor.py",
    transfer_queue_adapter,
]
missing_tq_paths = [path for path in required_tq_paths if not path.exists()]
if missing_tq_paths:
    missing = "\n  - ".join(str(path) for path in missing_tq_paths)
    raise SystemExit(
        "This NeMo RL image does not include the TransferQueue GRPO data-plane "
        "trainer required for the Polar rollout actor integration. Missing:\n"
        f"  - {missing}\n"
        "Use a NeMo RL image built from a revision that contains grpo_sync.py, "
        "sync_rollout_actor.py, and nemo_rl.data_plane. Do not runtime-overlay "
        "NeMo source into the official v0.6.0 image for this smoke."
    )


def rewrite(path: Path, transform) -> bool:
    original = path.read_text()
    updated = transform(original)
    if updated == original:
        return False
    path.write_text(updated)
    return True


def patch_grpo_sync(text: str) -> str:
    old_import = "from nemo_rl.experience.sync_rollout_actor import SyncRolloutActor\n"
    new_import = """if os.environ.get("NEMO_RL_POLAR_ROLLOUT", "0") == "1":
    from nemo_bridge.rollout_actor import PolarSyncRolloutActor as SyncRolloutActor

    _SYNC_ROLLOUT_ACTOR_FQN = "nemo_bridge.rollout_actor.PolarSyncRolloutActor"
else:
    from nemo_rl.experience.sync_rollout_actor import SyncRolloutActor

    _SYNC_ROLLOUT_ACTOR_FQN = "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor"
"""
    if new_import not in text:
        if old_import not in text:
            raise RuntimeError("Could not find SyncRolloutActor import in grpo_sync.py")
        text = text.replace(old_import, new_import)

    old_fqn = '"nemo_rl.experience.sync_rollout_actor.SyncRolloutActor"'
    if "make_actor_runtime_env(_SYNC_ROLLOUT_ACTOR_FQN)" not in text:
        text = text.replace(
            f"make_actor_runtime_env(\n            {old_fqn}\n        )",
            "make_actor_runtime_env(_SYNC_ROLLOUT_ACTOR_FQN)",
        )
        text = text.replace(
            f"make_actor_runtime_env({old_fqn})",
            "make_actor_runtime_env(_SYNC_ROLLOUT_ACTOR_FQN)",
        )

    old_mask = 'torch.ones_like(driver_carry["total_reward"])'
    new_mask = 'driver_carry["loss_multiplier"].to(driver_carry["total_reward"].device)'
    if new_mask not in text:
        if old_mask not in text:
            raise RuntimeError("Could not find GRPO baseline valid-mask expression")
        text = text.replace(old_mask, new_mask, 1)
    return text


def patch_registry(text: str) -> str:
    old_entry = '    "nemo_bridge.rollout_actor.PolarSyncRolloutActor": PY_EXECUTABLES.VLLM,\n'
    old_entry_vllm_exec = '    "nemo_bridge.rollout_actor.PolarSyncRolloutActor": VLLM_EXECUTABLE,\n'
    entry = '    "nemo_bridge.rollout_actor.PolarSyncRolloutActor": PY_EXECUTABLES.SYSTEM,\n'
    text = text.replace(old_entry, entry)
    text = text.replace(old_entry_vllm_exec, entry)
    if entry in text:
        return text
    anchor = (
        '    "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor": '
        "PY_EXECUTABLES.VLLM,\n"
    )
    if anchor not in text:
        raise RuntimeError("Could not find SyncRolloutActor registry entry")
    return text.replace(anchor, anchor + entry)


def patch_nvml(text: str) -> str:
    old = """def get_free_memory_bytes(device_idx: int) -> float:
    \"\"\"Get the free memory of a CUDA device in bytes using NVML.\"\"\"
    global_device_idx = device_id_to_physical_device_id(device_idx)
    with nvml_context():
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(global_device_idx)
            return pynvml.nvmlDeviceGetMemoryInfo(handle).free
        except pynvml.NVMLError as e:
            raise RuntimeError(
                f\"Failed to get free memory for device {device_idx} (global index: {global_device_idx}): {e}\"
            )
"""
    new = """def get_free_memory_bytes(device_idx: int) -> float:
    \"\"\"Get the free memory of a CUDA device in bytes.

    GB10 systems can expose limited NVML memory telemetry.  Prefer NVML when it
    works, but allow a torch.cuda.mem_get_info fallback for Spark smoke runs.
    \"\"\"
    if os.environ.get("NEMO_RL_NVML_MEM_GET_INFO_FALLBACK", "1") == "1":
        try:
            import torch

            free, _ = torch.cuda.mem_get_info(device_idx)
            return float(free)
        except Exception:
            pass

    global_device_idx = device_id_to_physical_device_id(device_idx)
    with nvml_context():
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(global_device_idx)
            return pynvml.nvmlDeviceGetMemoryInfo(handle).free
        except pynvml.NVMLError as e:
            raise RuntimeError(
                f\"Failed to get free memory for device {device_idx} (global index: {global_device_idx}): {e}\"
            )
"""
    if "torch.cuda.mem_get_info(device_idx)" in text:
        return text
    if old not in text:
        raise RuntimeError("Could not find get_free_memory_bytes body in nvml.py")
    return text.replace(old, new)


changed = {
    str(grpo_sync): rewrite(grpo_sync, patch_grpo_sync),
    str(registry): rewrite(registry, patch_registry),
    str(nvml): rewrite(nvml, patch_nvml),
}
for path, did_change in changed.items():
    print(f"{'patched' if did_change else 'already patched'} {path}")
PY

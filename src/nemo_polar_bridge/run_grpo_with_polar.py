"""Run NeMo's GRPO launcher with Polar as the async trajectory collector."""

from __future__ import annotations

import os
import runpy
from typing import Any

import nemo_rl.algorithms.async_utils as async_utils

from nemo_polar_bridge.collector import PolarAsyncTrajectoryCollector


def _pin_virtual_cluster_placement() -> None:
    """Pin NeMo train/inference placement groups when Spark node IPs are provided."""

    train_node_ip = os.environ.get("NEMO_POLAR_TRAIN_NODE_IP")
    inference_node_ip = os.environ.get("NEMO_POLAR_INFERENCE_NODE_IP")
    if not train_node_ip and not inference_node_ip:
        return

    import nemo_rl.distributed.virtual_cluster as virtual_cluster

    original_placement_group = virtual_cluster.placement_group

    def pinned_placement_group(*args: Any, **kwargs: Any) -> Any:
        name = str(kwargs.get("name") or "")
        target_ip = None
        if name.startswith("grpo_train_cluster"):
            target_ip = train_node_ip
        elif name.startswith("grpo_inference_cluster"):
            target_ip = inference_node_ip
        if target_ip:
            resource_name = f"node:{target_ip}"
            kwargs["bundles"] = [
                {**bundle, resource_name: 0.001} for bundle in kwargs["bundles"]
            ]
            print(f"Pinning NeMo placement group {name} to {resource_name}")
        return original_placement_group(*args, **kwargs)

    virtual_cluster.placement_group = pinned_placement_group


def main() -> None:
    _pin_virtual_cluster_placement()
    async_utils.AsyncTrajectoryCollector = PolarAsyncTrajectoryCollector
    runpy.run_path("/opt/nemo-rl/examples/run_grpo.py", run_name="__main__")


if __name__ == "__main__":
    main()

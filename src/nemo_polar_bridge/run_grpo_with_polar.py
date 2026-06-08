"""Run NeMo's GRPO launcher with Polar as the async trajectory collector."""

from __future__ import annotations

import runpy

import nemo_rl.algorithms.async_utils as async_utils

from nemo_polar_bridge.collector import PolarAsyncTrajectoryCollector


def main() -> None:
    async_utils.AsyncTrajectoryCollector = PolarAsyncTrajectoryCollector
    runpy.run_path("/opt/nemo-rl/examples/run_grpo.py", run_name="__main__")


if __name__ == "__main__":
    main()

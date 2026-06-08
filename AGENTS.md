# Repository Guidelines

## Project Structure & Module Organization

Polar is a Python package under `src/` with two top-level packages: `polar` for rollout, gateway, runtime, platform, trajectory, config, CLI, and agent harness code; and `slime_bridge` for the Slime trainer integration. Tests live in `tests/` and are grouped by subsystem (`tests/gateway`, `tests/platform`, `tests/trajectory`, `tests/rollout`). The dashboard is a Vite/React app in `web/`. Example deployments live in `examples/`, reusable scripts in `scripts/`, and documentation assets in `assets/`.

## Build, Test, and Development Commands

- `uv venv --python 3.13 && source .venv/bin/activate`: create the recommended local environment.
- `uv pip install -e .`: install Polar in editable mode.
- `uv pip install -e ".[swebench]"`: install optional SWE-bench dependencies.
- `pytest`: run the Python test suite.
- `ruff check`: run Python lint checks configured by `pyproject.toml`.
- `polar serve_rollout -c examples/count_stars/topology.yaml`: start the rollout orchestrator for an example topology.
- `cd web && npm install && npm run build`: install and build the dashboard bundle.
- `cd web && npm run dev`: run the dashboard locally during frontend development.

## Coding Style & Naming Conventions

Python targets 3.11+ and uses Ruff with a 99-character line length. Use 4-space indentation, typed Pydantic models for structured API/config data, and `snake_case` for modules, functions, variables, and tests. Keep subsystem boundaries clear: gateway transforms belong in `src/polar/gateway/transform/`, agent presets in `src/polar/agent/presets/`, and topology/config logic in `src/polar/config/` or `src/polar/platform/config.py`. React components use `PascalCase` filenames under `web/src/components/`; route views live under `web/src/routes/`.

## Testing Guidelines

Add focused pytest coverage near the subsystem changed. Name files `test_<behavior>.py` and tests `test_<expected_behavior>`. Use existing markers when relevant: `unit`, `integration`, and `gpu`. For server, gateway, runtime, or trajectory changes, also exercise the relevant `polar` CLI path or example topology.

## Commit & Pull Request Guidelines

Recent history uses short, descriptive commit titles, often with a PR number, for example `Polar Observability Dashboard (#32)` or `Fix pyproject (#22)`. Keep commits atomic and scoped. Pull requests should include motivation, changed modules, validation commands, linked issues, and dashboard screenshots when `web/` UI changes are visible.

## Security & Configuration Tips

Do not commit secrets, local logs, generated rollout results, trainer checkpoints, or bulky runtime artifacts. Keep topology examples reviewable and portable; document external services, GPU requirements, or patched inference-server assumptions in the relevant example README.

## NeMo/Polar RL Lab Status

Treat NeMo-only and Polar/ProRL claims separately. The two-Spark NeMo native
Async GRPO smoke has passed with Qwen3-0.6B using the pinned CUDA 13.2 NeMo main
image from `scripts/smoke/build_nemo_async_image.sh` and
`scripts/smoke/run_nemo_async_grpo_spark_smoke.sh`. That validates NeMo native
async, replay-buffer staleness, importance-sampling correction, non-colocated
vLLM generation, and train/inference weight sync across the Sparks.

Polar disaggregation has also passed as an external rollout proof: `spark-f7e2`
ran Polar rollout/gateway and calculator runtime sessions, while `spark-cfd0`
served Qwen3-0.6B through vLLM. A grouped calculator task (`num_samples=3`)
captured prompt tokens, response tokens, rollout logprobs, loss masks, rewards,
and result files for every session. The grouped Polar result was then validated
inside the NeMo image against `ReplayBufferImpl.add/sample`,
`add_grpo_token_loss_masks_and_generation_logprobs`, and
`batched_message_log_to_flat_message`.

Do not restore the old Polar -> NeMo TransferQueue patch path as the scalable
async engine. The remaining unproven gate is live NeMo trainer consumption of
external Polar groups. The next allowed code is the smallest possible collector
hook that feeds NeMo's native `ReplayBuffer` with Polar tokens, rollout
logprobs, rewards, `loss_multiplier`/`sample_mask`, and staleness metadata.

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

Treat NeMo-only and Polar/ProRL data-plane claims separately. We proved a NeMo-only two-Spark, non-colocated GRPO smoke with `nvcr.io/nvidia/nemo-rl:v0.6.0`, Qwen3-0.6B, one train step, training on `spark-f7e2`, and vLLM inference on `spark-cfd0`. We did **not** prove the end-to-end Polar/ProRL -> NeMo TransferQueue GRPO pipeline on the official prebuilt image.

The blocker is structural: official `v0.6.0` lacks `nemo_rl/algorithms/grpo_sync.py`, `nemo_rl/experience/sync_rollout_actor.py`, and `nemo_rl/data_plane/`. Do not runtime-overlay NeMo source into that image to claim compatibility. For Polar/ProRL integration, use a pinned official NeMo RL release image built from a revision whose installed source already contains those data-plane files. The current clean path is `scripts/smoke/build_nemo_polar_tq_image.sh`, which builds NVIDIA-NeMo/RL from a pinned CUDA 13.2 main revision, then validates CUDA, imports, vLLM, actor virtualenvs, TransferQueue, and GB10 capability before any Spark training smoke.

The Polar bridge contract must preserve rollout tokens, response logprobs, reward grouping, `sample_mask`/`loss_multiplier`, and train/inference weight-sync semantics. Failed Polar sessions must be excluded through `loss_multiplier`/`sample_mask`, not by NeMo's default `torch.ones_like(total_reward)` grouping mask.

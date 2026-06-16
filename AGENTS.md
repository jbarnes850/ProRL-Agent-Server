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

The live NeMo + Polar gate has now passed via
`scripts/smoke/run_nemo_polar_external_collector_spark_smoke.sh` with run dir
`/home/jarrodbarnes/nemo-rl-smoke/nemo-polar-qwen3-0p6b-live-20260607-222704`.
That run pinned NeMo training to `spark-cfd0` (`192.168.100.11`) and NeMo vLLM,
Polar rollout/gateway, and the Polar collector to `spark-f7e2`
(`192.168.100.10`). It completed one NeMo native Async GRPO step from two
Polar calculator attempts with rewards `[1.0, 0.0]`, nonzero group reward std,
token logprobs, token loss masks, replay-buffer add/sample, GRPO advantages,
policy training, and weight-sync pause/resume around Polar generation.

Do not restore the old Polar -> NeMo TransferQueue patch path as the scalable
async engine. The approved path is NeMo native Async GRPO plus the minimal
`nemo_polar_bridge` external collector feeding NeMo's native `ReplayBuffer`.

## Experiment-as-Code Engine (`nemo_polar_bridge.experiment`)

A declarative experiment-as-code layer over the approved Async GRPO + Polar
path. An `ExperimentSpec` (YAML) compiles to the NeMo base config plus dotted
overrides, registers as a content-addressed lineage asset, and emits a runnable
launch command. The whole layer is no-GPU and reuses the proven smoke launcher
rather than reimplementing orchestration.

    python -m nemo_polar_bridge.experiment.cli launch \
      examples/experiments/qwen3-1p7b-basic_arith-grpo-8x4.yaml \
      --store runs/lineage --out runs/<id>

- `experiment/spec.py` — `ExperimentSpec` (model/dataset/verifier/algorithm/
  async/rollout/topology/precision).
- `experiment/compile.py` — `compile_spec` -> `(base_config, dotted overrides)`.
  The algorithm axis is config-only on the GRPO family's single `ClippedPGLossFn`:
  `grpo` (baseline), `cispo` (clip `(1,4)` = Laguna parity), `drgrpo`
  (`grpo.normalize_rewards=false`, keeps the leave-one-out baseline), `gspo`,
  `dapo`, `rloo`; advantage via `grpo.adv_estimator.name`. Static guards: async
  excludes DAPO data-side features; `gdpo` needs a multi-reward verifier;
  `drgrpo` requires the `grpo` advantage. Baseline uses standard (unweighted)
  RLOO, not Laguna's length-weighted LOO (documented limitation).
- `experiment/lineage.py` — spec digest (identity) + injective compiled digest
  (what ran) + bidirectional DAG; forward artifact binding by content hash.
- `experiment/runtime.py` — `RuntimeProfile.two_spark()` + `compose_launch`. Spec
  knobs map to the smoke env contract; algorithm-axis keys the smoke does not set
  become positional `EXTRA_OVERRIDES`. Invariant: the GRPO baseline composes to
  zero positional extras.
- `scripts/experiment/validate_against_nemo_loader.py` — no-GPU dry-run gate:
  applies compiled overrides through NeMo RL's real `load_config` /
  `parse_hydra_overrides` (`struct=True`). Run inside the pinned image before any
  Spark launch to catch clone-vs-image key drift.

Key mappings were verified against NeMo RL `c236061b`; re-verify inside the
pinned image before a live run. The engine never spends GPU; the emitted
`launch.sh` is gated by the staged-validation ladder. Green baseline:
`PYTHONPATH=src .venv/bin/python -m pytest tests/nemo_polar_bridge/ -q`.

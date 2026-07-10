# Qwen3.5-9B RL Infrastructure Handoff

Use this as the source-of-truth brief for the Codex that will orchestrate the
actual RL workload. The infrastructure is optimized and stopped. No services or
containers should be assumed live.

## Final Verdict

- Standalone text-only Qwen3.5-9B FP8 inference at 131,072 context is stable on
  one DGX Spark.
- The production-shaped 65K-prefix, k=8, 1,536-token decode workload is stable
  and prefix-cache dominated.
- The two-Spark disaggregated NeMo/Polar path executes BF16 LoRA training,
  FP8 rollout, replay, trainer logprobs, policy update, cache invalidation, and
  optimized refit.
- Exact rollout-time top-p support replay is operational at `top_p=0.95`.
- Learning is not proven. Retained training smokes had zero within-group reward
  and advantage variance by design.
- Self-compaction artifact preservation is operational, but the first 116K
  semantic resume failed and the current GRPO/CISPO trainer rejects segmented
  trajectories.
- NVFP4/QARL and Modal were not made operational.

Confidence: high for infrastructure and serving throughput; unknown for
learning quality and long-sequence trainer fit.

## Hardware and Placement

| Role | Host | Direct IP | Precision |
|---|---|---|---|
| inference / Polar rollout | `spark-f7e2` | `192.168.100.10` | FP8 weights, KV auto |
| NeMo trainer | `spark-cfd0` | `192.168.100.11` | BF16 DTensor-v2 LoRA |

RoCE uses `enp1s0f0np0`, HCA `rocep1s0f0`, GID index 3. Each Spark has one GB10
with approximately 121 GiB unified memory. Do not tensor-parallelize the 9B
server across the two hosts.

## Optimal Standalone vLLM Configuration

```bash
docker run -d \
  --name qwen35-fp8-131k-prefix-seq8-bt65536 \
  --gpus all --network host --ipc=host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e VLLM_USE_DEEP_GEMM=0 \
  -v /home/jarrodbarnes/models:/models:ro \
  -v /home/jarrodbarnes/.cache/qwen35-vllm:/root/.cache/vllm \
  vllm/vllm-openai:v0.24.0-aarch64 \
  --model /models/Qwen/Qwen3.5-9B \
  --served-model-name qwen3.5-9b \
  --host 0.0.0.0 --port 8000 \
  --language-model-only \
  --quantization fp8 \
  --max-model-len 131072 \
  --kv-cache-dtype auto \
  --gpu-memory-utilization 0.35 \
  --max-num-batched-tokens 65536 \
  --max-num-seqs 8 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --enable-chunked-prefill \
  --max-logprobs 20 \
  --generation-config vllm
```

Rollback:

```bash
docker stop --timeout 30 qwen35-fp8-131k-prefix-seq8-bt65536
docker rm qwen35-fp8-131k-prefix-seq8-bt65536
```

Selected image: `vllm/vllm-openai:v0.24.0-aarch64`, image ID
`sha256:730a973ed3917e4eb96cb5c3a195272fe2712d291d86001ceba2f91053f41d4e`.

Measured production-shaped cell:

| Metric | Result |
|---|---:|
| prompt | 65,000 tokens/request |
| group | 8 requests sharing an identical prefix |
| decode | 1,536 tokens/request; 12,288 total |
| group wall time | 224.446 s |
| generated throughput | 54.748 tok/s |
| prefix-cache hit rate | 87.4246% |
| cold request | 121.901 s |
| warm maximum | 102.528 s |
| KV-cache capacity | 691,340 tokens / 5.27 full 131K contexts |
| FP8 speedup over matched BF16 | 1.367× |
| OOM / preemption | none / none |

The throughput lever is shared-prefix scheduling. Keep each prompt group's eight
rollouts adjacent so seven requests reuse the first request's prefill.

## NeMo/Polar RL Configuration

Start from:

- `examples/experiments/profiles/two_spark_qwen35_9b_safe.yaml`
- `examples/experiments/qwen3p5-9b-reasoning_gym-cispo-fp8-rollout-1x8-smoke.yaml`
- exact top-p cell:
  `examples/experiments/qwen3p5-9b-reasoning_gym-cispo-fp8-top-p095-replay-smoke.yaml`

Trainer:

- BF16 DTensor v2, TP1, CP1;
- LoRA rank 32, alpha 32, zero dropout;
- targets `model.language_model.*proj*`, covering attention, MLP, and GDN text
  projections;
- vision and audio towers frozen;
- activation checkpointing enabled;
- full fine-tuning is a no-fit path on one Spark.

Rollout:

- FP8 vLLM weights, `kv_cache_dtype=auto`;
- text-only, 131,072 context, `max_num_seqs=8`;
- prefix caching, aligned Mamba cache, chunked prefill;
- keep `max_num_batched_tokens=4096` for the exact retained integration proof;
  65,536 is the standalone vLLM optimum, not a proven embedded-vLLM setting;
- use the production decode cap of 1,536 for the real workload. The retained
  integration smoke used 256 only to prove the step/refit path.

Refit and transport:

- `NRL_QWEN35_FP8_REFIT=1`;
- `NRL_REFIT_SKIP_OPT_OFFLOAD=1` on the pinned c236 runtime;
- `NRL_REFIT_BUFFER_MEMORY_RATIO=0.01`;
- `NRL_FP8_BROADCAST=0`;
- `VLLM_USE_DEEP_GEMM=0`;
- ignore only `linear_attn.in_proj_a` and `linear_attn.in_proj_b` in FP8 because
  the GB10 CUTLASS padded 32-output shape fails;
- apply Qwen3.5's vLLM `WeightsMapper` before FP8 module classification;
- fail closed if aggregate rewrites are zero or any required module is
  unresolved;
- invalidate the prefix/KV cache after every policy-version refit.

Retained FP8 baseline: weight sync 4.35 s, total step 80.96 s. Retained exact
top-p replay: weight sync 3.46 s, total step 70.05 s.

## Exact Top-p Replay

Do not use naive `top_p<1`. FP8 rollout and BF16 trainer logits can select
different nuclei even when their maximum logit delta is only `8.01e-5`.

The operational contract records, for every trainable token:

- complete rollout-time `sampling_support_token_ids`;
- `sampling_support_lengths` and captured mass;
- sampled-token membership;
- rollout generation logprob and BF16 trainer recomputation over the exact same
  support.

The `top_p=0.95`, cap-256 live cell passed with 2,048/2,048 support coverage and
membership, generation KL error 0.0022, and 2.197% importance ratios outside
`[0.8, 1.2]`. The cap remains fail-closed; do not assume 256 is complete for a
new task distribution.

## Validation Modes and Learning Gates

Use `NRL_VALIDATION_MODE=infrastructure` only for plumbing. Set
`NRL_VALIDATION_MODE=learning` before interpreting gradients. Learning mode
must fail when:

- more than 50% of key rollout groups have reward standard deviation below
  `1e-5`;
- advantages are all zero;
- tokens, masks, rollout/trainer logprobs, replay add/sample, policy update,
  refit, or cleanup evidence is missing.

The existing runs prove the steps execute. They do not prove Qwen3.5 learns.
The trainer smoke also used a 4K sequence budget; the next operator must run a
safe long-sequence LoRA memory ladder before claiming that 37K/65K prompts train
on one Spark.

## Self-Compaction Boundary

`compaction_segments` preserves explicit execution → summary → execution
segments and records token counts, completion IDs, hashes, policy version, and
summary text. The live smoke compressed 116,167 tokens to 133 and resumed in
203 tokens without OOM, but lost the pending 0.97 action state.

The current NeMo GRPO/CISPO collector rejects these trajectories. Do not remove
that guard. Trainable self-compaction needs segment-aware PPO/critic or another
justified method with token-level loss normalization and cross-segment temporal
credit. See `self-compaction-protocol.md` and
`examples/compaction/qwen35-long-context-smoke.yaml`.

## Rejected or Deferred Paths

- FP8 KV cache: not the default; keep KV auto.
- `--enforce-eager`: rollback/debug lever, not the selected profile.
- FP8 broadcast: disabled; Qwen3.5 equivalence not proven.
- SGLang: no measured advantage over the retained vLLM profile in this program.
- AutoModel NVFP4: wrong compressor. ModelOpt W4A16 is the plausible future
  path.
- NeMo real-quant QARL: requires Megatron quantization-aware policy, not the
  current BF16 DTensor trainer.
- W4A4: later research only; do not conflate with W4A16 serving.
- Modal trainer: not implemented or required at this stop point.

## First Checks for the Training Codex

1. Run both-host health and confirm no live workloads before launching.
2. Recompose the experiment and inspect emitted `vllm_kwargs`; constructor-only
   scheduling knobs must not sit only under `vllm_cfg`.
3. Set the real 1,536 decode cap and run a BF16-LoRA sequence-length ladder.
4. Select a task mix with nonzero within-group reward variance.
5. Use `top_p=1.0` first, or enable exact support replay for any truncated
   sampling cell.
6. Inspect raw trajectories and the mechanical `validation_summary.json`; never
   infer success from process exit alone.
7. Stop immediately on a new kernel `NV_ERR_NO_MEMORY`, even if Docker reports
   `OOMKilled=false`.

Detailed evidence lives in `program.md`, `results.json`, `serving_matrix.csv`,
`stage3.md`, `top-p-replay-parity.md`, and `self-compaction-protocol.md`.

# Qwen3.5-9B Spark + NeMo/Polar RL Program

Status: stopped after infrastructure and throughput optimization
Started: 2026-07-10
Goal thread: `019f4c9f-4ef4-7a21-a6bf-112d4e8debe1`

## Decision

Determine whether Qwen3.5-9B can become the two-DGX-Spark lab's reproducible
scientific-RL model without sacrificing host safety, rollout/trainer probability
parity, or trajectory provenance.

The active completion target for this session is durable disaggregated
infrastructure plus the highest stable grouped-prefix throughput at the
131,072-token serving contract. Learning quality, reward variance, advantage
variance, and verifier improvement belong to the separately orchestrated
training session. They remain recorded telemetry here but are not acceptance
gates. The user stopped the program after the Stage 5 non-training artifact
smoke; QARL and Modal are not operational requirements for this handoff.

This training-step smoke is worth doing because it will improve the decision of
whether the lab operator can start the separately orchestrated Qwen3.5-9B LoRA
run, as measured by serving headroom, complete trajectory artifacts, replay,
trainer logprob recomputation, optimizer invocation, versioned FP8 refit, and
clean teardown, producing a reproducible infrastructure go/no-go report.

No training-step smoke begins until the standalone serving, logprob,
configuration-loader, and trajectory-schema gates pass. Reward and advantage
statistics are captured but do not block an infrastructure smoke.

## Scope And Posture

- Preserve the dirty worktree. The pre-existing changes to
  `scripts/smoke/build_nemo_async_image.sh` and
  `scripts/smoke/build_nemo_mopd_image.sh` are out of scope.
- Stage 1 is configuration-only. Source edits are allowed only after a named gate
  proves that current vLLM, NeMo, Polar, or experiment-spec surfaces cannot
  express the required contract.
- One single-Spark serving process per candidate. Do not tensor-parallelize the
  9B server across Sparks.
- CUDA is containerized. Host `nvcc` is not a gate.
- Start at 4,096 tokens, one sequence, 4,096 batched tokens, and memory
  utilization 0.35.
- The serving contract floor is 131,072 tokens. Lower-context points are safety
  gates only and cannot be selected as the final profile.
- Keep `kv_cache_dtype=auto` for the baseline and RL path. FP8 KV cache remains a
  separately labeled comparison, never the default.
- Qwen3.5-9B is multimodal, but this program is text-only. The kept vLLM baseline
  must use `--language-model-only`, reject image/video inputs, avoid vision-tower
  allocation/warmup, and still return generated token IDs plus token logprobs.
- Start with `top_p=1.0`. No top-p learning claim is allowed until trainer
  logprobs use the same truncated sampling distribution as rollout.
- Stage 5 is limited to a non-training segmented-artifact smoke. Stage 6 QARL
  and Stage 7 Modal remain source designs only and are not launched.

## Workload Contract

- Phase prompts are prefill-dominated: approximately 37K tokens p50 and 65K
  tokens p95, with generation capped at 1,536 JSON tokens.
- Temporal prompts are approximately 3-5K tokens.
- Each record produces a k=8 rollout group with an identical prompt prefix.
- A 65,536-token profile excluded 12 valid episodes, so 131,072 is a structural
  environment requirement rather than a stretch target.
- Throughput selection must therefore optimize grouped long-prefill work, not
  short-request decode throughput. Required measurements are cold prefill,
  seven warm prefix-sharing requests, aggregate group wallclock, cached-token
  count/hit rate, decode throughput, and token/logprob parity.

## Evidence Status Labels

- `confirmed`: live evidence on the intended runtime and model.
- `proxy`: relevant evidence from another model, backend, or historical config.
- `blocked`: the required path cannot run under current prerequisites.
- `contradicted`: source or runtime evidence rejects the claim.
- `unknown`: not yet tested.

## Source Baselines

- Local repo HEAD at start: `235e6bbb` on branch `weight-sync-reduction`.
- Proven Qwen3-4B path: the checked-in FP8/BF16 GRPO and shell-harness reports
  under `docs/goals/` plus their archived `runs/*/evidence` artifacts.
- Proven rollback image: `local/nemo-rl-main-cu132:c236061b`.
- Selected Stage 3 image is the proven fork-stable runtime:
  `local/nemo-rl-main-cu132:c236061b`, NeMo RL
  `c236061b250e97638722292ab8a54d5eb47ae00f`.
- Latest official NeMo RL inspected on 2026-07-10:
  `6ab16882addfcfc778f6f6bee21454edda9487ff`; the only later commit adds
  Megatron MFU reporting and does not invalidate the Stage 3 image pin.
- Standalone vLLM image present on both Sparks:
  `vllm/vllm-openai:v0.24.0-aarch64`, image
  `sha256:730a973ed3917e4eb96cb5c3a195272fe2712d291d86001ceba2f91053f41d4e`.
- vLLM image source revision:
  `ee0da84ab9e04ac7610e28580af62c365e898389`.
- Latest official vLLM inspected on 2026-07-10:
  `c227aaa3f8edd02dae4583e27246430eebabfb25`.

## Stage Gates

| Stage | Required evidence | Current status |
|---|---|---|
| 0. Recovery | Both hosts healthy and idle; identical checkpoint; image and OOM audit; recovery report | confirmed |
| 1. Minimal serving | Text-only 4K/1/4096/0.35/auto server; completion; token IDs; logprobs; 10 repeats; clean stop | confirmed |
| 2. Fit ladder | One-axis context through mandatory 131K, then k=8 concurrency/prefix-cache ladder with long-prefill workload metrics | confirmed; FP8 selected at 54.748 tok/s |
| 3. NeMo/Polar baseline | Real config loader; tiny live rollout; executable LoRA step; complete artifact audit; versioned refit; clean teardown | confirmed in BF16 and FP8; learning validation intentionally fails |
| 4. Top-p parity | Latest NeMo filtering audit; rollout/trainer support parity; mismatch, `-inf`, KL, IS telemetry | confirmed at `top_p=0.95`; exact rollout support replay passed live infrastructure audit |
| 5. Self-compaction | Multi-turn rubric and schema first; pre-summary/resume traces; verifier and cost evidence | segmented artifacts passed at 116,167 tokens; semantic resume reliability failed; training no-go |
| 6. QARL | ModelOpt W4A16 frozen serving first; probability/task parity; then latest NeMo Megatron packed real-quant refit; W4A4 only after W4A16 | not operational; source audit only per user stop |
| 7. Spark rollout + Modal trainer | Strict versioned delta/trajectory contract | not started per user stop |

## Iteration Rules

1. Change one axis at a time and append one row to `serving_matrix.csv`.
2. Capture exact image, command, model hashes, timestamps, request payloads,
   response artifacts, host memory before/after, and container logs.
3. Keep a candidate only when endpoint readiness, token IDs, logprobs, repeated
   requests, host health, and clean stop all pass.
4. On one OOM, discard the config and reduce context, concurrency, or memory
   utilization immediately. On two OOMs on one axis, stop that axis. On five
   consecutive discards, pause the program and write the failure synthesis.
5. Infrastructure validation treats reward and advantage variance as telemetry.
   Learning validation requires nonzero advantages and fails when more than 50%
   of groups have reward standard deviation below `1e-5`. Missing tokens,
   logprobs, masks, replay events, optimizer invocation, versioned weight sync,
   placement, or cleanup are hard failures in either mode.

## vLLM Optimization Audit

The retained vLLM 0.24 startup logs prove that the old launch omitted text-only
controls: it allocated a 16,384-token multimodal encoder-cache budget and ran
multimodal warmup. The v0.24 source already exposes `--language-model-only`; its
multimodal config sets all modality limits to zero, the registry selects
text-only mode, and tower modules are skipped at construction. This is the first
optimization to test because it removes irrelevant vision cost without changing
the language model's sampling distribution.

Other candidates remain gated:

- Keep compilation/CUDA graphs enabled initially; the historical servers were
  stable with them and graph memory was below 1 GiB. `--enforce-eager` is a
  rollback lever, not the default performance profile.
- Do not enable FP8 KV cache in the baseline. Its accuracy and RL parity are a
  separate experiment.
- Do not enable speculative decoding unless the exact Qwen3.5-9B checkpoint
  contains a compatible draft/MTP contract and token/logprob parity is re-proven.
- Enable prefix caching for the standalone grouped-rollout benchmark after the
  131K cold-cache fit gate. Qwen3.5 is a hybrid GDN/attention architecture, so
  use the supported aligned Mamba cache mode rather than `all`. Before prefix
  caching can enter the RL baseline, prove cache invalidation on every weight
  version update and show no stale-logit/logprob reuse.

## Disaggregated Stage 3 Contract

- Trainer Spark: Qwen3.5-9B policy and optimizer in BF16.
- Rollout Spark: vLLM model weights in FP8; `kv_cache_dtype=auto`.
- Non-colocated NCCL refit over RoCE with no trainer optimizer offload.
- `top_p=1.0`, importance-sampling correction enabled, and trajectory age 1.
- Text-only Qwen3.5 engine with 131,072 model context, eight sequences,
  4,096 batched tokens for the integration gate, prefix caching, aligned Mamba
  cache, and chunked prefill.
- The infrastructure smoke keeps the 131,072-token engine contract and uses a
  256-token generation cap so coherent completion, trainer consumption, and
  post-step refit are observable. Throughput benchmarking separately uses the
  workload cap of 1,536 generated JSON tokens.

Standalone vLLM uses the measured FP8 65,536-batched-token production profile.
NeMo's embedded vLLM 0.20 keeps the conservative 4,096-token integration chunk;
the standalone throughput choice does not need to become the trainer smoke
shape.

The prior BF16-rollout launch was a wiring probe only. It was stopped before a
rollout or policy update when the live vLLM log proved that
`max_num_batched_tokens=65536` had been placed under `vllm_cfg`; NeMo only
forwards this engine constructor argument from `vllm_kwargs`, so the engine used
its 2,048-token default. The compiler now emits both concurrency controls under
`policy.generation.vllm_kwargs`, and the retained experiment declares FP8
rollout weights explicitly.

## Stage 4 Exact Top-p Replay

The pinned NeMo runtime's native top-p path reconstructs the nucleus from BF16
trainer logits. That is not exact when FP8 rollout logits select a different
support. The implemented path instead records vLLM's processed rollout-time
support per sampled token, preserves it through Polar and replay, and computes
differentiable BF16 trainer logprobs over those exact token IDs.

The retained `top_p=0.95` r3 run passed the mechanical infrastructure audit:
2,048/2,048 trainable tokens had complete support, every sampled token belonged
to its support, the policy step executed, and optimized FP8 refit completed in
3.46 seconds. End-to-end throughput was 17.02 generated tok/s/GPU and generation
KL error was 0.0022. The run had zero within-group reward variance, so it remains
explicitly invalid as a learning result. Full evidence and the cap-selection
probe are in `top-p-replay-parity.md`.

Exact support replay is currently admitted only for TP1, CP1, unpacked
sequences, and a fail-closed support cap. The learning session owns the
`top_p=0.98/0.95/0.90` entropy and verifier sweep.

## Stage 5 Self-Compaction Boundary

Polar now exposes a `compaction_segments` builder with explicit summary/resume
markers, typed execution/summary segments, completion lineage, server-tokenized
costs, policy version, and token hashes. The reserved marker is captured in the
original request and stripped before inference. Unmarked prompt rewrites fail
closed.

The non-training r1 smoke used a 116,167-token pre-compaction context, produced
a 133-token model-authored summary, and resumed in 203 tokens. All three traces
and logprobs were preserved; the 131K FP8 server stopped cleanly without a new
OOM. The state-retention verifier failed because a pending 0.97 refinement was
converted into a completed action and then omitted from the resumed answer.

The NeMo collector deliberately rejects these segmented trajectories. Current
Async CISPO/GRPO cannot correctly assign credit across variable numbers of
compaction segments. No self-compaction training claim is made. See
`self-compaction-protocol.md`.

## Stage 6 NVFP4 Boundary

Status: on hold. This section preserves the completed source audit only; it
does not authorize serving experiments, refit work, trainer-topology changes,
or GPU spend in the current program.

The detailed source decision is recorded in `nvfp4-source-audit.md`.
AutoModel supports the dense Qwen3.5 architecture but is not the current NVFP4
checkpoint compressor. ModelOpt main now ships a dense Qwen3.5 W4A16-MSE recipe,
and the retained vLLM 0.24 image selects both native W4A4 and Marlin W4A16
kernels on GB10. This makes a frozen serving ablation feasible, not validated.

The humans& result is Qwen3-30B-A3B MoE at 8K with MoE-only NVFP4,
dequantized BF16 backward, weight-and-activation four-over-six, and selective
BF16 layers. Its SGLang online quantizer explicitly does not support dense
linears. Its embedded data show close mean reward but a 3.26x larger
train/rollout absolute logprob difference than BF16. It is therefore evidence
for a controlled ablation, not for near-zero-loss replacement of the FP8
baseline.

The admitted first candidate is ModelOpt W4A16-MSE with the stock recipe's FP8
KV-cache unit removed so runtime KV remains `auto`. It must first pass static
131K text-only serving, 65K x k=8 prefix-sharing throughput, token/logprob KL,
importance-ratio, and verifier-parity gates. Live NVFP4 refit is later: current
NeMo RL implements it only through a Megatron quantization-aware policy with
BF16 master/backward and quantized forward. DTensor quantization is explicitly
unimplemented, so this cannot enter the Stage 3 BF16-DTensor path as a config
change.

## Required Deliverables

- `program.md` (this file)
- `recovery.md`
- `results.json`
- `serving_matrix.csv`
- exact best standalone vLLM command and rollback command
- NeMo/Polar baseline run directory and artifact audit
- top-p replay parity report
- self-compaction protocol and failed resume-reliability evidence
- Stage 4 go/no-go for starting the actual Qwen3.5-9B Spark training workload

## Current Best Standalone Command

This is the confirmed Stage 2 standalone optimum:

```bash
docker run -d \
  --name qwen35-text-131k-auto-035-prefix-seq8-bt65536 \
  --gpus all --network host --shm-size 16g \
  -v /home/jarrodbarnes/models/Qwen/Qwen3.5-9B:/models/Qwen/Qwen3.5-9B:ro \
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
  --max-logprobs 20
```

Rollback:

```bash
docker stop --timeout 30 qwen35-text-131k-auto-035-prefix-seq8-bt65536
docker rm qwen35-text-131k-auto-035-prefix-seq8-bt65536
```

## Blocked Stop Condition

Stop and report rather than expand scope when host safety, exact image/model
compatibility, text-only logprob validity, reward-bearing data, trainer/rollout
parity, or versioned recovery cannot be proven under the allowed surfaces. The
report must list attempted configurations, raw evidence, the blocker, and the
smallest next input or authorization required.

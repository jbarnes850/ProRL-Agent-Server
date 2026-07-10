# Stage 3: Disaggregated NeMo/Polar Baseline

Status: BF16 and FP8 LoRA infrastructure passed; learning gate deferred
Topology: one BF16 trainer Spark plus one FP8 vLLM/Polar rollout Spark
KV cache: `auto`

## Source Decision

The tagged NeMo RL v0.6.0 release is not the correct baseline for this run. It
ships vLLM 0.17.1 and predates the required Qwen3.5/refit behavior. Latest main
was source-audited, but the live baseline is the already proven fork-stable
`c236061b250e97638722292ab8a54d5eb47ae00f` image plus the isolated
Qwen3.5 FP8 mapper patch. This avoided rebuilding the live baseline around
unrelated latest-main changes.

The published Spark recipe's main result has landed upstream. Commit
`a45aa533c0920d742c165de76702326caed8ab99` removes
`policy.offload_before_refit()` and the matching trainer restore from the
non-colocated vLLM collective path. This is the native form of the recipe's
226.5s-to-1.73s refit fix, so the latest-main candidate must not mount the custom
skip patch.

Other upstream changes included in the candidate:

- `4f7b4aa8`: merges FP8 quantization overrides with user HF overrides instead
  of replacing them. This matters because the 131K profile may carry model
  config overrides.
- `79c38c2f`: explicit environment, generation-worker, and policy-worker
  shutdown ordering.
- `892e54eb`: avoids an expensive full-response logging path for NeMo Gym. The
  external Polar/reasoning-gym smoke still retains the mandatory artifact audit.
- `9e01af64`: NUMA-aware binding. Both Sparks expose one socket and one NUMA
  node, so this is expected to be neutral rather than a throughput lever.
- `f2ab62c2`: fused chunked Megatron logprobs avoid materializing full
  `[batch, sequence, vocabulary]` logits. This is potentially important for
  65K/131K trainer sequences, but it is Megatron-only and therefore a later
  backend ablation, not a change to the BF16 DTensor baseline.

Latest main still pins vLLM 0.20.0 internally. The standalone optimum remains
vLLM 0.24.0; upgrading NeMo's embedded vLLM independently would mix two axes and
is blocked until the pinned-main FP8 integration smoke passes.

The live RoCE device is `rocep1s0f0` with netdev `enp1s0f0np0` and IPv4 RoCEv2
GID index 3 on both hosts. The launcher's older `rocep1s0f1` default is down;
the Qwen3.5 runtime profile now pins the complete HCA/netdev/GID tuple.

## Aborted Wiring Probe

The first non-colocated process-start probe used BF16 rollout weights and the
proven 131K serving shape. Both the Qwen3.5 DTensor trainer and the text-only
vLLM worker loaded without OOM. It was deliberately stopped before rollout or
training because the engine announced a 2,048-token chunk size rather than
65,536.

Root cause: the smoke launcher placed `max_num_seqs` and
`max_num_batched_tokens` under `policy.generation.vllm_cfg`. NeMo accepts those
keys in Hydra, but constructs the vLLM engine from free-form
`policy.generation.vllm_kwargs`; the values were therefore inert.

No RL result is claimed from that run. It proves process placement and model
load only.

Current main also removed the Qwen3-specific SGLang base recipe used by the
older launcher. Both c236 and current main contain
`examples/configs/grpo_math_1B_sglang.yaml`; the compiler and launcher now use
that shared base. The real NeMo loader accepts all 40 compiled overrides on both
revisions.

## Corrected Experiment

Spec:
`examples/experiments/qwen3p5-9b-reasoning_gym-cispo-fp8-rollout-1x8-smoke.yaml`

Required effective values before GPU launch:

| Surface | Value |
|---|---:|
| `policy.precision` | `bfloat16` |
| `policy.generation.vllm_cfg.precision` | `fp8` |
| `policy.generation.vllm_cfg.kv_cache_dtype` | `auto` |
| `policy.generation.vllm_cfg.max_model_len` | `131072` |
| `policy.generation.vllm_kwargs.max_num_seqs` | `8` |
| `policy.generation.vllm_kwargs.max_num_batched_tokens` | `4096` |
| `policy.generation.vllm_kwargs.language_model_only` | `true` |
| `policy.generation.vllm_kwargs.mamba_cache_mode` | `align` |
| `policy.generation.vllm_kwargs.enable_chunked_prefill` | `true` |
| `policy.generation.vllm_cfg.enable_prefix_caching` | `true` |
| `policy.generation.top_p` | `1.0` |
| `loss_fn.use_importance_sampling_correction` | `true` |
| `policy.generation.colocated.enabled` | `false` |

The 4,096-token integration value is deliberate. Standalone vLLM retains the
65,536-token throughput profile, while NeMo's embedded vLLM 0.20 keeps the
conservative chunk proven by the end-to-end LoRA smoke. It does not reduce the
131,072 model-context contract.

The compiler now routes the two constructor-only scheduling values through
`vllm_kwargs`. The Polar collector also accepts current-main's empty optional
distillation metadata, rejects nonempty teacher metadata, and implements the
new neutral efficiency-metrics interface. Because replacing NeMo's collector
also replaced NeMo's post-refit hook, the Polar hook now invalidates vLLM's
prefix/KV cache before gateway resume whenever in-flight recomputation is
enabled; failed or partial invalidation leaves the gateway paused and fails the
run. Targeted compiler/runtime/collector/refit tests pass 46/46 and Ruff is clean.

## Historical FP8 Refit Failure (Resolved)

The serving engine starts safely on `spark-f7e2` with 131,072 context, eight
sequences, 16,384 batched tokens, FP8 weights, automatic KV dtype, aligned
prefix caching, and CUDA graphs. The retained startup reported 222,288 KV
tokens and no OOM. The BF16 LoRA trainer loaded on `spark-cfd0`, froze
vision/audio, and patched all 248 language-model projection linears while
excluding vision and `lm_head`.

That first attempt was not a valid RL baseline. The collector exposed
that every response was multilingual gibberish with token logprobs near a
uniform distribution. The server had been loaded with dummy weights and NeMo
printed `Could not find module` for every refit parameter while still reporting
the refit as successful. The resulting k=8 rewards were all zero, so the run was
stopped at the reward-variance gate before accepting a policy update.

Root cause is source-proven. NeMo sends HF keys such as
`model.language_model.layers.*`; vLLM's Qwen3.5 internal Python modules are
`language_model.model.layers.*`, with `lm_head.*` under
`language_model.lm_head.*`. NeMo's FP8 pre-classifier does not apply vLLM's
multi-segment `WeightsMapper` prefixes, so it misses the real text linears.

Two adapter probes were discarded:

| Run | Result | Decision |
|---|---|---|
| `...lora32-text-refit-r3` | Existing overlay lacked latest-main `bind_numa`; failed before refit | discard; overlay compatibility failure |
| `...lora32-text-refit-r4` | Startup passed; fail-closed gate saw `target=CUDAGraphWrapper`, `rewritten=0`; refit aborted before rollout | discard; wrapper-aware mapping required |

At that checkpoint the source-only adapter was rebased to the exact vLLM mapping:

- `model.language_model.* -> language_model.model.*`
- `lm_head.* -> language_model.lm_head.*`
- frozen vision weights are not forwarded to the text-only rollout worker

It accumulates rewritten counts across refit buckets and fails when the total
is zero. The subsequent clean fork-stable run satisfied the live refit gate:
nonzero mapped weights, coherent text, finite aligned token logprobs, a
versioned post-step refit, and clean resource recovery.

Learning remains outside this infrastructure session. The retained task mix
produced zero within-group reward variance and is rejected by learning-mode
validation.

## Go/No-Go Gate

This session proves infrastructure, not learning. Reward, advantage, entropy,
and verifier statistics are retained for the next training session and do not
block this gate. An infrastructure pass requires that the exact LoRA path can
roll out, replay, recompute trainer logprobs, invoke an optimizer step, advance
the rollout weight version, invalidate stale cache state, and tear down cleanly.

The fork-stable BF16 control passed this complete path on 2026-07-10 with a
256-token generation cap: 16 trajectories, aligned tokens and logprobs, replay
add/sample, trainer policy call, generation KL error 0.0002, weight version
`0 -> 1`, post-step refit, and clean two-host resource recovery. One group was
uniformly rewarded 0 and one uniformly rewarded 1, so advantages were zero;
that is a learning-quality observation, not an infrastructure failure.

The retained FP8 infrastructure step passed all of these gates:

1. The selected image reports its exact source ref, CUDA 13, Torch 2.11, GB10
   compute capability 12.1, and imports every Ray actor environment.
2. The no-GPU loader's final merged config shows BF16 policy, FP8 vLLM,
   automatic KV dtype, non-colocation, 131K context, eight sequences, and
   4,096 batched tokens in their effective locations.
3. Live vLLM startup logs show FP8 quantization, text-only Qwen3.5, 131K model
   length, prefix caching, and the 4,096-token integration chunk.
4. Tokens and generation logprobs align in the Polar response.
5. The tiny run produces masks, rewards, advantages, replay add/sample, policy
   update, weight version/refit timing, and clean teardown without OOM.

Any FP8 initialization failure, silent BF16 fallback, 2,048-token chunk,
missing logprob, or host pressure is a discard, not a result.

## Optimized Refit Resolution

The Qwen3.5 mapper patch resolves the initial FP8 failure without replacing
NeMo's vLLM backend. It applies vLLM's native `WeightsMapper` before FP8 module
classification and returns `None` for unresolved modules. Initial and
post-step FP8 refits now produce coherent text with no missing-language-module
warnings.

The report's production setting is now explicit:
`NRL_REFIT_SKIP_OPT_OFFLOAD=1`, `NRL_REFIT_PROFILE=1` for diagnostics,
`NRL_REFIT_BUFFER_MEMORY_RATIO=0.01`, RoCE, and
`NRL_FP8_BROADCAST=0`. Every setting is recorded in `config.json`.

Matched 2x8, 256-token diagnostic cells:

| Metric | BF16 rollout | FP8 rollout |
|---|---:|---:|
| Weight sync | 3.82s | 4.35s |
| Total step | 81.08s | 80.96s |
| E2E tok/s/GPU | 21.96 | 21.95 |
| Generation KL error | 0.0002 | 0.0041 |
| IS clip fraction | 0.0746% | 2.7674% |
| Post-step producer profile | 3.05s | 3.53s |
| Post-step consumer profile | 2.72s | 3.24s |

The skip reduced BF16 sync from 6.77s to 3.82s and FP8 sync from 11.96s to
4.35s. The 131x full-policy result from the Spark report does not transfer
numerically to LoRA because the rank-32 optimizer state is small; the setting is
still correct for the non-colocated topology.

Both runs pass `validation_mode=infrastructure` and fail
`validation_mode=learning`: both groups have reward standard deviation below
`1e-5`, advantages are zero, and loss is zero. This session makes no learning
claim. The launcher now requires an explicit `NRL_VALIDATION_MODE` contract.
This profile pins `infrastructure`; the next training session must override it
to `learning`. Learning mode adds `--require-learning-signal`, fails when more
than 50% of groups are near-zero variance or when advantages remain zero, and
propagates the audit's nonzero exit status to the launcher instead of printing
`SUCCEEDED`.

The earlier unoptimized FP8 `infra-r4` run is discarded even though its
driver exited zero: the post-run kernel audit found
`NV_ERR_NO_MEMORY` on the trainer at 16:26 and inference host at 16:28.
The optimized `r6` run supersedes it and produced no later driver-memory
event.

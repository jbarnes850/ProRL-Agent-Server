# Stage 4: Exact Top-p Sampling-Distribution Replay

Status: infrastructure parity passed for `top_p=0.95`
Audited: 2026-07-10
Live run: `/home/jarrodbarnes/nemo-rl-smoke/qwen35-9b/qwen35-forkstable-c236-fp8-lora32-top-p095-replay-r3-20260710-2250`

## Decision

The BF16-LoRA trainer/FP8-rollout path can now train against the exact
rollout-time top-p support. The retained `top_p=0.95` infrastructure run passed
the complete contract: vLLM captured the processed nucleus, Polar preserved it,
replay serialized it, NeMo recomputed differentiable trainer logprobs only over
that support, the policy step executed, FP8 refit completed, and the mechanical
auditor accepted the resulting artifacts.

This is a systems-validity result, not evidence that top-p improves learning.
Entropy, reward, advantage, pass-rate, and verifier comparisons remain owned by
the separate learning session. The infrastructure smoke intentionally used a
task group with zero reward variance, so its loss and advantages are zero.

## Why Native NeMo Recomposition Was Insufficient

NeMo RL revisions v0.6.0 (`5fb58893`), the selected runtime (`c236061b`), and
latest main (`6ab16882`) all reconstruct top-p support from trainer logits.
They do not serialize the rollout-time kept set. That distinction matters when
rollout uses FP8 weights and training uses BF16 weights: a maximum logit delta
of only `8.0108642578125e-05` changed the nucleus in the pinned runtime probe.
The native mismatch path then replaced an out-of-support `-inf` with zero
without removing the token from the caller's loss mask.

The implementation therefore does not reuse NeMo's locally reconstructed
nucleus. It carries the actual rollout-time token IDs and uses them as the
trainer normalization domain.

## Implemented Contract

`sampling_support_token_ids` is token-aligned with `token_ids`:

- prompt and non-trainable positions carry empty support;
- every trainable sampled token carries its complete rollout-time kept-token
  ID set;
- every sampled token must belong to its recorded support;
- `sampling_support_lengths` distinguishes the ragged support from padding;
- `sampling_support_mass` must be numerically complete;
- run metadata records top-p, temperature, support cap, precision, image,
  NeMo revision, refit flags, and topology.

The trainer gathers current BF16 logits at the recorded IDs and computes the
restricted log-normalizer there. The normalizer remains differentiable. The
path fails closed for missing/incomplete support, sampled-token exclusion,
out-of-range IDs, TP/CP other than 1, or packed sequences.

Source surfaces:

- `src/polar/gateway/engine.py`: request processed top-logprobs and reconstruct
  complete per-token support.
- `src/polar/trajectory/` and `src/nemo_polar_bridge/collector.py`: preserve and
  serialize support through Polar and replay.
- `src/nemo_polar_bridge/sampling_support.py`: patch the pinned c236 train-data
  constructors and install support-aware prev/current logprob computation.
- `scripts/smoke/audit_nemo_polar_run.py`: require complete coverage and sampled
  membership when `top_p < 1` replay is enabled.
- `src/nemo_polar_bridge/experiment/spec.py`: reject unlabeled naive top-p
  configurations.

## Support-Cap Probe

The cap was selected empirically before the live run. The probe used vLLM's
processed rollout distribution and rejected any token whose reconstructed mass
was incomplete.

| Cap | Incomplete positions | Minimum captured mass | Artifact size | Probe time |
|---:|---:|---:|---:|---:|
| 20 | 40 / 768 | 0.8068 | 3.05 MB | 11.524 s |
| 64 | 11 / 768 | 0.9477 | 8.75 MB | 11.570 s |
| 128 | 3 / 768 | 0.9858 | 16.02 MB | 11.648 s |
| 256 | 0 / 768 | 0.9999998353 | 29.56 MB | 11.789 s |

At cap 256 the probe's largest support was 172 tokens, mean support was 5.22,
and p95 was 21. The live arithmetic smoke was sharper: mean 1.38, p95 2, and
maximum 256 across 2,048 trainable positions. Its minimum recorded mass was
`0.9999998808`; the cap therefore remained complete under the fail-closed
tolerance, but a production task mix must continue to retain the completeness
gate rather than assume 256 is universally sufficient.

## Live Evidence

The retained r3 run used Qwen3.5-9B, BF16 LoRA rank 32 training, FP8 rollout,
`kv_cache_dtype=auto`, `top_p=0.95`, support cap 256, RoCE refit, and
`NRL_REFIT_SKIP_OPT_OFFLOAD=1`.

| Gate | Result |
|---|---:|
| Docker/NeMo exit | 0 |
| Mechanical validation | passed |
| Trainable support coverage | 2,048 / 2,048 (100%) |
| Sampled-token membership | 2,048 / 2,048 (100%) |
| Hidden `-inf` support failure | none observed |
| Policy update | executed |
| Weight sync | 3.46 s |
| Total step | 70.05 s |
| End-to-end throughput | 17.02 generated tok/s/GPU |
| Generation KL error | 0.0022 |
| Mean `prev / generation` ratio | 0.99747 |
| Ratio standard deviation | 0.06839 |
| Ratio outside `[0.8, 1.2]` | 2.197% |
| Wallclock | 304 s |

The run had one rollout group, all rewards zero, advantage standard deviation
zero, and loss zero. `validation_mode=infrastructure` records this honestly and
does not treat it as a learning result. The same artifact fails the learning
variance gate because 100% of groups have reward standard deviation below
`1e-5`.

Both Sparks were clean after teardown: no live containers, no occupied lab
service ports, and no new kernel OOM or `NV_ERR_NO_MEMORY` entries during the
run window.

## Failed Attempts And Resolution

- r1 failed before model load because the fresh deployment omitted the
  Qwen3.5 FP8 weight mapper. It caused no OOM and cleanup passed.
- r2 completed the support-aware rollout, training, and refit, but the pinned
  c236 logger omitted the support fields from its audit artifact. The launcher
  correctly classified the run as failed.
- r3 changed only evidence serialization. It passed without weakening the
  support, training, or cleanup gates.

## Remaining Learning Matrix

| Cell | Purpose | Status |
|---|---|---|
| `top_p=1.0` | untruncated control | Stage 3 infrastructure passed |
| `top_p=0.95`, native recompute | negative control | structurally mismatched; do not use for claims |
| `top_p=0.95`, exact support replay | systems parity | passed |
| `top_p=0.98/0.95/0.90` | entropy/reward sweep | learning session only |

The next training session may use exact support replay, but it must still pick
a task mix where at most 50% of rollout groups have near-zero reward variance
before interpreting gradients or learning metrics.

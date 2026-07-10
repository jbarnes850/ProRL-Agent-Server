# Stage 5: Self-Compaction Protocol and Infrastructure Gate

Status: segmented artifact path passed; resume reliability failed; training no-go
Audited: 2026-07-10
Live run: `/home/jarrodbarnes/qwen35-compaction/stage5-r1`

## Verdict

Polar can now preserve a real Qwen3.5-9B context rewrite without pretending it
is an append-only chat. The live non-training run captured the pre-compaction
execution segment, model-authored summary segment, resumed execution segment,
token/logprob arrays, completion lineage, policy version, hashes, token costs,
and a read-only verifier result. The 131K FP8 server stayed healthy and stopped
cleanly.

The model did not pass the semantic resume gate. It changed a pending next
calculation—refine oxygen occupancy constrained to 0.97—into a completed action,
then omitted 0.97 from the resumed answer. That is a genuine lost-state error.
The system must preserve this episode as failed evidence and must not train on
it through the current NeMo GRPO/CISPO collector.

## Decision and Rubric

Capability axis: long-horizon scientific state preservation and recovery.

Stakeholder decision: whether the next training operator can admit compacted
scientific trajectories without silently losing experimental state or
corrupting credit assignment.

Primary rubric:

1. The task's read-only verifier result is unchanged or improved relative to an
   uncompressed continuation with the same action access.
2. Exact identifiers, measurements, units, accepted/rejected hypotheses,
   reasons, constraints, pending actions, and unresolved errors survive.
3. The resumed policy can use those fields rather than merely repeat them.
4. Every rewrite is replayable from raw completions and token-level lineage.
5. Environment/verifier failures are distinct from model lost-state failures.
6. Cost is reported separately: source, summary, resumed-context, generated,
   tool, and wall-clock totals.

The future training justification is:

> This training run is worth doing because it will improve the training
> operator's decision to admit compacted scientific trajectories, as measured
> by integrity-clean verifier parity, lost-state rate, resume success, and cost,
> producing a model-authored replayable state handoff through a multi-turn
> environment where the policy can execute, summarize, resume, use tools, and
> revise.

That sentence does not authorize training today because the segment-aware
optimizer and reward-bearing task corpus are absent.

## Turn Structure

Environment turn structure: multi-turn search/revision.
Episode horizon: original task horizon, with zero to three compactions.
State after each turn: immutable raw request/response, tool observation,
server-tokenized IDs/logprobs, policy version, token/tool/time costs.
Policy actions: ordinary execution/tool action, explicit summary action,
resumed execution, final answer/escalation.
Tool/verifier access: scientific tools remain policy-visible; the final
scientific verifier remains read-only and evaluator-only.
Observation after calls: real tool result or fail-closed environment error.
Reward timing: terminal task reward, with turn-level diagnostic fields only.
Credit assignment unit: future segment-aware token objective, never naive
per-segment reward duplication.
Same-action-space baselines: no-compaction at the same peak context and
inference-time compaction with the same tools.
Posthoc diagnostics: full-context replay oracle and field-level lost-state
audit; neither is a policy action.

## Implemented Schema

The `compaction_segments` builder requires explicit `_polar_compaction`
metadata in the original API request. The marker is captured but stripped
before forwarding to vLLM/SGLang. Heuristic prefix-break classification is
rejected because an unexplained rewrite may be truncation, harness drift, or
corruption.

Each event records:

- summary and resume trace indices;
- source, summary, and resume completion IDs;
- actual server-tokenized trigger count and context/threshold contract;
- recent-turn retention setting;
- pre-compaction, summary, and resumed-context token counts;
- summary text and compression ratio;
- SHA-256 hashes for the pre-compaction tokens, summary tokens, and resumed
  context;
- policy/weight version.

Each trace is typed as `execution` or `summary` and retains ordinary token IDs,
loss masks, rollout logprobs, messages, and top-p support when enabled.

The current NeMo collector rejects any trajectory containing
`compaction_events`. It does not concatenate a rewritten prompt into an
ordinary response stream.

## Live Evidence

The smoke used vLLM 0.24 text-only Qwen3.5-9B, FP8 weights,
`kv_cache_dtype=auto`, 131,072 context, memory utilization 0.35, prefix caching,
aligned Mamba cache, chunked prefill, and compiled CUDA graphs.

| Measurement | Result |
|---|---:|
| Session status | completed |
| Segment sequence | execution → summary → execution |
| Pre-compaction context | 116,167 tokens |
| Summary | 133 tokens |
| Summary/source ratio | 0.11449% |
| Resumed context | 203 tokens |
| Gateway run time | 74.643 s |
| Reported aggregate prefix-cache hit rate | 49.7% |
| Reported prompt throughput window | 11,662.6 tok/s |
| KV-cache capacity | 676,948 tokens |
| Full-131K concurrency estimate | 5.16× |
| New OOM/driver-memory events | none |
| Teardown | clean; no containers or lab ports remained |
| Read-only state-retention reward | 0.0 |

The summary preserved the sample ID, phase candidate, pressure, rejected
candidate, rejection residual, and the value 0.97. It corrupted action status:
the 0.97 refinement was pending but became “Completed Action.” The resumed JSON
preserved the other fields and omitted 0.97. This is `model_error/lost_state`,
not an environment timeout.

The original r1 verifier used exact prose substrings and therefore also emitted
false negatives for semantically preserved fields. The reusable smoke now uses
field-level checks and a dedicated pending-next-action line check. The retained
r1 reward remains unchanged; evidence is never rescored silently.

Raw completion persistence intentionally omitted the redundant response
`prompt_text` and `kv_transfer_params` after the per-field 1 MiB cap. Original
requests, response token IDs/logprobs, the complete 1.41 MiB trajectory result,
and all compaction hashes were preserved. Production archival should continue
to treat the terminal trajectory plus original requests as authoritative.

## Why Training Stops Here

Cognition describes self-compaction as asking the policy to summarize near the
context limit, resuming from that self-authored state, and alternating
success-only phases with weighted token/turn/tool-time budget phases
([SWE-1.7](https://cognition.ai/blog/swe-1-7)).

CompactionRL makes the optimizer mismatch explicit: variable segment counts
make group-wise methods ill-suited, while independent segment training biases
loss weighting and temporal credit. Its implementation moves to PPO with a
critic, token-level loss normalization, and cross-trajectory GAE
([CompactionRL](https://arxiv.org/html/2607.05378v1)). The current lab path is
native Async CISPO/GRPO with group-normalized advantages. Reusing it would be a
method change disguised as plumbing.

Slipstream further shows why summary fluency is not validation: independently
continued pre-compaction behavior supplies a trajectory-grounded check on
forward intent and required facts
([Slipstream](https://arxiv.org/html/2605.08580)). That is the right future
validation baseline, but it is not implemented in this stop point.

## Admission Gates for Any Future Training Session

1. Segment-aware PPO/critic path or another explicitly justified
   cross-segment credit method.
2. Token-level normalization that prevents segment-count and segment-length
   weighting bias.
3. Cross-boundary temporal credit and policy-version provenance.
4. Real scientific tasks with read-only verifiers and an uncompressed
   same-action-space comparator.
5. Field-level lost-state taxonomy: omission, commission, status inversion,
   unit corruption, stale state, and unsupported next action.
6. Environment/verifier error rate at or below 5%, zero high-severity integrity
   exploits, and deterministic reset/replay.
7. Reward-bearing smoke before GPU training; group-normalized methods still
   require at most 50% near-zero within-group reward variance.
8. Only after reliability parity: alternate unconstrained success phases with
   budget phases penalizing weighted tokens, turns, and tool time.

## Sources

- [SWE-1.7: Frontier Intelligence at a Fraction of the Cost](https://cognition.ai/blog/swe-1-7) (2026-07-10)
- [CompactionRL: Reinforcement Learning with Context Compaction for Long-Horizon Agents](https://arxiv.org/html/2607.05378v1) (2026-07-06)
- [Slipstream: Trajectory-Grounded Compaction Validation for Long-Horizon Agents](https://arxiv.org/html/2605.08580) (2026-05-12)

Parallel search output: `/tmp/qwen35-self-compaction.json`.

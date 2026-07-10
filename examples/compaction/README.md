# Qwen3.5 Long-Context Compaction Smoke

`qwen35-long-context-smoke.yaml` is a non-training Polar task that exercises a
real context rewrite against a Qwen3.5-9B endpoint. It sends an approximately
116K-token scientific working state, asks the policy to author a compact
summary, resumes from that summary, and applies a deterministic read-only state
retention verifier.

Requirements:

- a Polar topology whose gateway serves Qwen3.5-9B with at least 131,072
  context;
- the `compaction_segments` trajectory builder;
- the `polar-spark-calculator:latest` runtime image, or an equivalent image with
  Python 3;
- no trainer. The current NeMo GRPO/CISPO collector deliberately rejects these
  segmented trajectories.

Run through the normal Polar CLI:

```bash
polar submit -c topology.yaml --json examples/compaction/qwen35-long-context-smoke.yaml
```

Passing proves artifact preservation and state retention for this synthetic
probe. It does not prove that trainable self-compaction improves a scientific
task. Inspect `trajectory.compaction_events`, all three traces, evaluator
details, and gateway completion records before admitting a task family.

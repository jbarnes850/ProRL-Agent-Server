# NeMo + Polar GRPO

This example is the minimal Polar external-rollout surface for NeMo RL GRPO.
NeMo owns policy training, TransferQueue, logprob recomputation, and vLLM
weight sync. Polar owns async agent/session rollout through its gateway.

This requires a NeMo RL image whose installed source already contains
`nemo_rl/algorithms/grpo_sync.py`, `nemo_rl/experience/sync_rollout_actor.py`,
and `nemo_rl/data_plane/`. The official `nvcr.io/nvidia/nemo-rl:v0.6.0`
image does not expose that data-plane trainer path, so it is valid for
NeMo-only legacy GRPO smoke tests but not for this adapter.
Build the pinned CUDA 13.2 main image with
`scripts/smoke/build_nemo_polar_tq_image.sh`; the example YAML defaults to
`local/nemo-rl-main-cu132:37526dfa`.

1. Start Polar with `topology.yaml`, updating `gateway.nodes[0].inference.base_url`
   to the OpenAI-compatible vLLM endpoint exposed by NeMo.
2. Patch NeMo in the training container:
   `NEMO_RL_POLAR_ROLLOUT=1 scripts/patch/patch_nemo_polar.sh`.
3. Merge `nemo_polar_fragment.yaml` into a NeMo GRPO config, or pass equivalent
   Hydra overrides for the top-level `polar` block.

The default agent is a shell harness that makes one chat-completion request via
the Polar gateway. It is intentionally small so token ids, rollout logprobs,
sample masks, reward grouping, and weight-sync semantics can be validated before
adding heavier agent harnesses.

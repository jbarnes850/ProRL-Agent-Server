from nemo_polar_bridge.refit import adapt_qwen35_language_only_weights


def test_adapts_composite_qwen35_weights_for_causal_lm_target() -> None:
    weights = [
        ("model.visual.patch_embed.proj.weight", "vision"),
        ("model.language_model.layers.0.mlp.up_proj.weight", "up"),
        ("model.language_model.embed_tokens.weight", "embed"),
        ("lm_head.weight", "head"),
    ]

    adapted, rewritten, dropped = adapt_qwen35_language_only_weights(
        weights,
        target_module_names=["model.layers.0.mlp.up_proj", "model.embed_tokens"],
    )

    assert adapted == [
        ("model.layers.0.mlp.up_proj.weight", "up"),
        ("model.embed_tokens.weight", "embed"),
        ("lm_head.weight", "head"),
    ]
    assert rewritten == 2
    assert dropped == 1


def test_adapts_hf_names_for_vllm_conditional_generation_target() -> None:
    weights = [
        ("model.visual.patch_embed.proj.weight", "vision"),
        ("model.language_model.layers.0.mlp.up_proj.weight", "up"),
        ("model.language_model.embed_tokens.weight", "embed"),
        ("lm_head.weight", "head"),
    ]

    adapted, rewritten, dropped = adapt_qwen35_language_only_weights(
        weights,
        target_module_names=[
            "visual.patch_embed.proj",
            "language_model.model.layers.0.mlp.up_proj",
            "language_model.model.embed_tokens",
            "language_model.lm_head",
        ],
    )

    assert adapted == [
        ("language_model.model.layers.0.mlp.up_proj.weight", "up"),
        ("language_model.model.embed_tokens.weight", "embed"),
        ("language_model.lm_head.weight", "head"),
    ]
    assert rewritten == 3
    assert dropped == 1


def test_is_noop_for_composite_qwen35_target() -> None:
    weights = [("model.language_model.layers.0.mlp.up_proj.weight", "up")]

    adapted, rewritten, dropped = adapt_qwen35_language_only_weights(
        weights,
        target_module_names=["model.language_model.layers.0.mlp.up_proj"],
    )

    assert adapted == weights
    assert rewritten == 0
    assert dropped == 0

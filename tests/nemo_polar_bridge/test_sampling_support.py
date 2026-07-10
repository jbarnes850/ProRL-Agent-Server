from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from nemo_polar_bridge.sampling_support import (  # noqa: E402
    install_nemo_worker_sampling_support_patch,
    patch_c236_grpo_for_sampling_support,
    restricted_support_next_token_logprobs,
)


def _fixture_logits() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.2, 1.2, -0.3, 0.7, -1.0],
                [0.4, -0.2, 1.1, 0.3, 0.0],
                [0.1, 0.5, -0.4, 1.3, 0.2],
            ]
        ],
        dtype=torch.float32,
        requires_grad=True,
    )


def test_restricted_support_uses_next_token_alignment_and_has_gradients() -> None:
    logits = _fixture_logits()
    input_ids = torch.tensor([[4, 1, 2, 3]])
    support_ids = torch.tensor(
        [[[-1, -1, -1], [1, 3, -1], [0, 2, 3], [1, 3, -1]]]
    )
    support_lengths = torch.tensor([[0, 2, 3, 2]])
    token_mask = torch.tensor([[0, 1, 1, 1]])

    actual = restricted_support_next_token_logprobs(
        logits,
        input_ids,
        support_ids,
        support_lengths,
        token_mask,
    )
    expected = torch.stack(
        [
            logits[0, 0, 1] - torch.logsumexp(logits[0, 0, [1, 3]], dim=0),
            logits[0, 1, 2] - torch.logsumexp(logits[0, 1, [0, 2, 3]], dim=0),
            logits[0, 2, 3] - torch.logsumexp(logits[0, 2, [1, 3]], dim=0),
        ]
    ).unsqueeze(0)

    torch.testing.assert_close(actual, expected)
    (-actual.sum()).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_restricted_support_scales_logits_by_temperature() -> None:
    logits = _fixture_logits()
    input_ids = torch.tensor([[4, 1, 2, 3]])
    support_ids = torch.tensor(
        [[[-1, -1], [1, 3], [2, 3], [1, 3]]]
    )
    lengths = torch.tensor([[0, 2, 2, 2]])
    mask = torch.tensor([[0, 1, 0, 0]])

    actual = restricted_support_next_token_logprobs(
        logits,
        input_ids,
        support_ids,
        lengths,
        mask,
        temperature=2.0,
    )
    scaled = logits[0, 0] / 2.0
    expected = scaled[1] - torch.logsumexp(scaled[[1, 3]], dim=0)
    torch.testing.assert_close(actual[0, 0], expected)
    assert actual[0, 1:].tolist() == [0.0, 0.0]


def test_restricted_support_rejects_missing_sampled_token() -> None:
    with pytest.raises(ValueError, match="sampled token is absent"):
        restricted_support_next_token_logprobs(
            _fixture_logits(),
            torch.tensor([[4, 1, 2, 3]]),
            torch.tensor([[[-1, -1], [0, 3], [2, 3], [1, 3]]]),
            torch.tensor([[0, 2, 2, 2]]),
            torch.tensor([[0, 1, 0, 0]]),
        )


def test_restricted_support_rejects_empty_active_support() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        restricted_support_next_token_logprobs(
            _fixture_logits(),
            torch.tensor([[4, 1, 2, 3]]),
            torch.tensor([[[-1, -1], [-1, -1], [2, 3], [1, 3]]]),
            torch.tensor([[0, 0, 2, 2]]),
            torch.tensor([[0, 1, 0, 0]]),
        )


def test_c236_grpo_patch_preserves_support_in_both_train_paths(tmp_path: Path) -> None:
    anchor = """                            "sample_mask": repeated_batch["loss_multiplier"],
                        }
                    )"""
    path = tmp_path / "grpo.py"
    sync_log = """                log_data["prev_logprobs"] = train_data["prev_logprobs"].tolist()

                logger.log_batched_dict_as_jsonl("""
    async_log = """            log_data["prev_logprobs"] = train_data["prev_logprobs"].tolist()
            logger.log_batched_dict_as_jsonl("""
    path.write_text(
        f"sync\n{anchor}\n{sync_log}\nasync\n{anchor}\n{async_log}\n"
    )

    assert patch_c236_grpo_for_sampling_support(path) is True
    patched = path.read_text()
    assert patched.count("nemo-polar: preserve rollout sampling support") == 2
    assert patched.count("train_data[support_key] = flat_messages[support_key]") == 2
    assert patched.count("log_data[support_key] = train_data[support_key].tolist()") == 2
    assert patch_c236_grpo_for_sampling_support(path) is False


def test_c236_grpo_patch_rejects_unknown_source(tmp_path: Path) -> None:
    path = tmp_path / "grpo.py"
    path.write_text("different revision\n")

    with pytest.raises(RuntimeError, match="expected two c236"):
        patch_c236_grpo_for_sampling_support(path)


def test_run_bridge_does_not_import_grpo_before_patch() -> None:
    sys.modules.pop("nemo_rl.algorithms.grpo", None)
    import nemo_polar_bridge.run_grpo_with_polar  # noqa: F401

    assert "nemo_rl.algorithms.grpo" not in sys.modules


def test_patch_applies_to_exact_installed_c236_grpo(tmp_path: Path) -> None:
    installed = Path("/opt/nemo-rl/nemo_rl/algorithms/grpo.py")
    if not installed.exists():
        pytest.skip("exact NeMo c236 source is available only in its container")
    copied = tmp_path / "grpo.py"
    copied.write_text(installed.read_text())

    assert patch_c236_grpo_for_sampling_support(copied) is True
    compile(copied.read_text(), str(copied), "exec")


def test_exact_nemo_worker_paths_use_recorded_support(monkeypatch) -> None:
    pytest.importorskip("nemo_rl")
    from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
    from nemo_rl.algorithms.loss.interfaces import LossInputType
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    import nemo_rl.models.automodel.train as automodel_train

    monkeypatch.setenv("NEMO_POLAR_REQUIRE_SAMPLING_SUPPORT_REPLAY", "1")
    original_prepare = automodel_train.prepare_loss_input
    processor_type = install_nemo_worker_sampling_support_patch()
    assert processor_type is not None

    logits = _fixture_logits()
    input_ids = torch.tensor([[4, 1, 2, 3]])
    support_ids = torch.tensor(
        [[[-1, -1, -1], [1, 3, -1], [0, 2, 3], [1, 3, -1]]]
    )
    support_lengths = torch.tensor([[0, 2, 3, 2]])
    token_mask = torch.tensor([[0, 1, 1, 1]])
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": torch.tensor([4]),
            "token_mask": token_mask,
            "sample_mask": torch.tensor([1.0]),
            "sampling_support_token_ids": support_ids,
            "sampling_support_lengths": support_lengths,
            "sampling_support_mass": torch.tensor([[0.0, 1.0, 1.0, 1.0]]),
        }
    )
    sampling_params = TrainingSamplingParams(top_p=0.95, temperature=1.0)

    processor = object.__new__(processor_type)
    processor.sampling_params = sampling_params
    processor.cp_size = 1
    processor.enable_seq_packing = False
    logprobs = processor(
        logits,
        data,
        SimpleNamespace(input_ids=input_ids),
        original_batch_size=1,
        original_seq_len=4,
    )
    assert logprobs.shape == (1, 4)
    assert logprobs[0, 0] == 0
    assert torch.isfinite(logprobs).all()

    class Loss:
        input_type = LossInputType.LOGPROB
        reference_policy_kl_penalty = 0.0
        use_linear_ce_fusion = False

    loss_input, _ = automodel_train.prepare_loss_input(
        logits,
        data,
        Loss(),
        sampling_params=sampling_params,
    )
    torch.testing.assert_close(loss_input["next_token_logprobs"], logprobs[:, 1:])
    (-loss_input["next_token_logprobs"].sum()).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    automodel_train.prepare_loss_input = original_prepare

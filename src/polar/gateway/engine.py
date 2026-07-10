"""Inference backend strategies for the Polar gateway.

The gateway speaks the OpenAI Chat Completions API to a local inference server.
Two backends are supported, and they differ only in:

  1. the request params that make them emit the token ids + per-token logprobs
     Polar needs for training, and
  2. the exact shape of those fields in the response.

The base implements the canonical contract -- request ``logprobs`` (the one
training param every backend needs) and a pass-through response. A backend
overrides only what it does differently, via two hooks: ``prepare_request``
(extra request params) and ``normalize_response`` (response canonicalization).
Everything downstream -- storage, trace builder, transforms, slime adapter --
then sees one shape. The canonical shape is Polar's training output:

  - prompt token ids:   ``choice.input_token_ids`` (or ``response.prompt_token_ids``)
  - response token ids: ``choice.token_ids``       (or ``logprobs.content[].token_id``)
  - per-token logprobs: ``choice.logprobs.content[]`` with ``{token, token_id, logprob, ...}``

SGLang reaches this shape via source-supported prompt-token/meta-info extensions
plus a light response normalization. vLLM reaches it natively via the
``return_token_ids`` request flag plus a light response rename.
"""

from __future__ import annotations

from abc import ABC
import math
import os
import re
from typing import Any


class InferenceEngine(ABC):
    """Strategy for one OpenAI-compatible inference backend.

    The base encodes the canonical contract: request ``logprobs`` (the one
    training param every backend needs) and pass the response through
    unchanged. A backend overrides only what it does differently.
    """

    name: str

    def prepare_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Inject the request params this backend needs to emit training signals.

        ``logprobs`` is universal; subclasses add backend-specific params (e.g.
        token-id flags) on top via ``super().prepare_request(...)``.
        """
        request["logprobs"] = True
        return request

    def normalize_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Canonicalize the backend's response (in place) and return it.

        The default is a pass-through; a backend that differs from Polar's
        training shape overrides this.
        """
        return response

    @staticmethod
    def _stamp_token_ids_onto_logprobs(choice: dict[str, Any]) -> None:
        """Copy sampled token IDs onto OpenAI chat logprob entries when aligned."""

        token_ids = choice.get("token_ids")
        logprobs = choice.get("logprobs")
        if not isinstance(token_ids, list) or not isinstance(logprobs, dict):
            return
        content = logprobs.get("content")
        if not isinstance(content, list) or len(content) != len(token_ids):
            return
        for entry, token_id in zip(content, token_ids):
            if isinstance(entry, dict):
                entry.setdefault("token_id", token_id)


class SGLangEngine(InferenceEngine):
    """SGLang via its source-supported OpenAI-compatible extensions."""

    name = "sglang"

    def prepare_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request = super().prepare_request(request)  # logprobs=True
        request["return_prompt_token_ids"] = True
        request["return_meta_info"] = True
        return request

    def normalize_response(self, response: dict[str, Any]) -> dict[str, Any]:
        choices = response.get("choices")
        if not isinstance(choices, list):
            return response
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            self._canonicalize_prompt_token_ids(choice)
            self._canonicalize_response_token_ids(choice)
            self._stamp_token_ids_onto_logprobs(choice)
            # return_meta_info is requested as an internal bridge to recover
            # token IDs from source SGLang; do not expose or store the duplicate
            # raw meta payload downstream.
            choice.pop("meta_info", None)
        return response

    @staticmethod
    def _canonicalize_prompt_token_ids(choice: dict[str, Any]) -> None:
        if choice.get("input_token_ids") is not None:
            return
        prompt_token_ids = choice.get("prompt_token_ids")
        if isinstance(prompt_token_ids, list):
            choice["input_token_ids"] = list(prompt_token_ids)

    @staticmethod
    def _canonicalize_response_token_ids(choice: dict[str, Any]) -> None:
        if isinstance(choice.get("token_ids"), list):
            return
        meta_info = choice.get("meta_info")
        if not isinstance(meta_info, dict):
            return
        output_logprobs = meta_info.get("output_token_logprobs")
        if not isinstance(output_logprobs, list):
            return

        token_ids: list[int] = []
        for item in output_logprobs:
            token_id = None
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                token_id = item[1]
            elif isinstance(item, dict):
                token_id = item.get("token_id")
            if token_id is None:
                continue
            try:
                token_ids.append(int(token_id))
            except (TypeError, ValueError):
                continue

        if token_ids:
            choice["token_ids"] = token_ids


class VLLMEngine(InferenceEngine):
    """vLLM via its native OpenAI-compatible server.

    ``return_token_ids`` makes vLLM emit ``response.prompt_token_ids`` and
    ``choice.token_ids``. `top_logprobs`` must be set (not None) for vLLM
    to populate ``logprobs.content[]`` given ``logprobs=True``; 0 returns just
    the sampled token's logprob, which is all training needs.
    """

    name = "vllm"

    @staticmethod
    def _sampling_support_enabled() -> bool:
        return os.environ.get("POLAR_VLLM_CAPTURE_SAMPLING_SUPPORT", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def prepare_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request = super().prepare_request(request)  # logprobs=True
        request["return_token_ids"] = True
        if self._sampling_support_enabled() and float(request.get("top_p", 1.0)) < 1.0:
            support_cap = int(os.environ.get("POLAR_VLLM_SAMPLING_SUPPORT_SIZE", "0"))
            if support_cap <= 0:
                raise ValueError(
                    "POLAR_VLLM_SAMPLING_SUPPORT_SIZE must be positive when "
                    "sampling-support capture is enabled with top_p < 1"
                )
            request["top_logprobs"] = support_cap
        else:
            request.setdefault("top_logprobs", 0)
        # vLLM reads input reasoning from `reasoning`, not Polar's canonical
        # `reasoning_content`; rename it so prior turns' interleaved thinking
        # survives templating (else they render an empty `<think></think>`).
        for message in request.get("messages") or []:
            if isinstance(message, dict) and message.get("reasoning_content") is not None:
                message["reasoning"] = message.pop("reasoning_content")
        return request

    def normalize_response(self, response: dict[str, Any]) -> dict[str, Any]:
        choices = response.get("choices")
        if not isinstance(choices, list):
            return response
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            self._canonicalize_reasoning(choice.get("message"))
            self._stamp_token_ids_onto_logprobs(choice)
            if self._sampling_support_enabled():
                self._extract_sampling_support(choice)
        return response

    @staticmethod
    def _top_logprob_token_id(item: dict[str, Any]) -> int | None:
        token_id = item.get("token_id")
        if token_id is not None:
            try:
                return int(token_id)
            except (TypeError, ValueError):
                return None
        token = item.get("token")
        if not isinstance(token, str):
            return None
        match = re.fullmatch(r"token_id:(\d+)", token)
        return int(match.group(1)) if match else None

    @classmethod
    def _extract_sampling_support(cls, choice: dict[str, Any]) -> None:
        logprobs = choice.get("logprobs")
        content = logprobs.get("content") if isinstance(logprobs, dict) else None
        if not isinstance(content, list):
            raise ValueError("vLLM response is missing token-aligned logprobs content")

        tolerance = float(os.environ.get("POLAR_VLLM_SAMPLING_SUPPORT_MASS_TOL", "1e-5"))
        for position, entry in enumerate(content):
            if not isinstance(entry, dict):
                raise ValueError(f"invalid vLLM logprob entry at position {position}")
            top_logprobs = entry.get("top_logprobs")
            if not isinstance(top_logprobs, list) or not top_logprobs:
                raise ValueError(
                    f"vLLM response is missing sampling support at position {position}"
                )

            support_ids: list[int] = []
            support_mass = 0.0
            for item in top_logprobs:
                if not isinstance(item, dict):
                    continue
                try:
                    logprob = float(item.get("logprob"))
                except (TypeError, ValueError):
                    continue
                # vLLM clamps filtered -inf values to -9999 in OpenAI responses.
                if not math.isfinite(logprob) or logprob <= -9000.0:
                    continue
                token_id = cls._top_logprob_token_id(item)
                if token_id is None:
                    raise ValueError(
                        "vLLM sampling support requires token-id responses; "
                        f"position {position} contained {item.get('token')!r}"
                    )
                support_ids.append(token_id)
                support_mass += math.exp(logprob)

            if not support_ids:
                raise ValueError(f"empty vLLM sampling support at position {position}")
            if abs(1.0 - support_mass) > tolerance:
                raise ValueError(
                    "incomplete vLLM sampling support at position "
                    f"{position}: mass={support_mass:.9f} tolerance={tolerance}"
                )

            sampled_token_id = entry.get("token_id")
            if sampled_token_id is None or int(sampled_token_id) not in support_ids:
                raise ValueError(
                    f"sampled token is outside recorded support at position {position}"
                )
            entry["sampling_support_token_ids"] = support_ids
            entry["sampling_support_mass"] = support_mass
            if os.environ.get("POLAR_VLLM_KEEP_RAW_TOP_LOGPROBS", "").strip().lower() not in {
                "1",
                "true",
                "yes",
                "on",
            }:
                entry.pop("top_logprobs", None)

    @staticmethod
    def _canonicalize_reasoning(message: Any) -> None:
        """vLLM names the field ``reasoning``; Polar's canonical field is ``reasoning_content``."""
        if not isinstance(message, dict):
            return
        if message.get("reasoning_content") is None and message.get("reasoning") is not None:
            message["reasoning_content"] = message.pop("reasoning")

    @staticmethod
    def _stamp_token_ids_onto_logprobs(choice: dict[str, Any]) -> None:
        """Parity with SGLang: copy token_id onto each logprob entry.

        vLLM builds ``logprobs.content`` and ``choice.token_ids`` from the same
        ``output.token_ids``, so they align; guard on equal length regardless.
        Not load-bearing for training (which reads ``choice.token_ids`` and the
        per-entry ``logprob``) -- it keeps stored traces one shape across engines.
        """
        InferenceEngine._stamp_token_ids_onto_logprobs(choice)


_ENGINES: dict[str, type[InferenceEngine]] = {
    SGLangEngine.name: SGLangEngine,
    VLLMEngine.name: VLLMEngine,
}


def get_engine(name: str) -> InferenceEngine:
    """Return the inference engine strategy for ``name`` (``sglang`` | ``vllm``)."""
    try:
        engine_cls = _ENGINES[name]
    except KeyError:
        supported = ", ".join(sorted(_ENGINES))
        raise ValueError(
            f"Unknown inference engine {name!r}; supported: {supported}"
        ) from None
    return engine_cls()

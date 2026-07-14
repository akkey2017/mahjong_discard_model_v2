"""Schema-checked atomic checkpoints for step-based ViT training."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from mahjong_ai.models import ViTConfig


CHECKPOINT_SCHEMA_VERSION = "vit-training-checkpoint-v1"
MODEL_FAMILY = "mahjong-snapshot-vit-multitask"


class CheckpointCompatibilityError(ValueError):
    pass


def checkpoint_module(model):
    return getattr(model, "_orig_mod", model)


def atomic_torch_save(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_payload(path: Path, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise CheckpointCompatibilityError("checkpoint is not a mapping")
    return payload


def validate_training_checkpoint(
    payload: dict,
    *,
    model_config: ViTConfig,
    feature_schema_version: str,
    target_schema_version: str,
    dataset_manifest_sha256: str,
) -> None:
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY,
        "feature_schema_version": feature_schema_version,
        "target_schema_version": target_schema_version,
        "dataset_manifest_sha256": dataset_manifest_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise CheckpointCompatibilityError(
                f"{key} mismatch: checkpoint={payload.get(key)!r}, expected={value!r}"
            )
    if payload.get("model_config") != model_config.to_dict():
        raise CheckpointCompatibilityError("ViTConfig differs from checkpoint")


def load_legacy_vit_weights(path: Path, model, map_location="cpu") -> dict:
    """Initialize from legacy ViT weights; never restores optimizer state."""

    payload = load_payload(path, map_location=map_location)
    state = payload.get("model_state", payload)
    if not isinstance(state, dict):
        raise CheckpointCompatibilityError("legacy checkpoint has no state_dict")
    state = {
        key.removeprefix("_orig_mod."): value
        for key, value in state.items()
        if torch.is_tensor(value)
    }
    single_head = "fc.weight" in state
    if single_head:
        state["heads.dapai.weight"] = state.pop("fc.weight")
        state["heads.dapai.bias"] = state.pop("fc.bias")
    result = checkpoint_module(model).load_state_dict(state, strict=not single_head)
    allowed_missing = {
        f"heads.{task}.{suffix}"
        for task in ("riichi", "fulou", "gang", "hule")
        for suffix in ("weight", "bias")
    }
    if single_head and (set(result.missing_keys) != allowed_missing or result.unexpected_keys):
        raise CheckpointCompatibilityError(
            f"unsupported legacy keys: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    return {
        "source_model_type": payload.get("model_type"),
        "single_head_migration": single_head,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }

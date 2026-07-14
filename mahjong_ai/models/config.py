"""Configuration and presets for the ViT-only model family."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class ViTConfig:
    name: str
    in_channels: int = 380
    embed_dim: int = 512
    depth: int = 8
    heads: int = 8
    patch_size: tuple[int, int] = (1, 1)
    dropout: float = 0.1
    drop_path: float = 0.1
    mlp_ratio: float = 4.0
    use_cls_token: bool = True

    def __post_init__(self) -> None:
        if self.in_channels < 1 or self.embed_dim < 1 or self.depth < 1 or self.heads < 1:
            raise ValueError("ViT dimensions must be positive")
        if self.embed_dim % self.heads:
            raise ValueError("embed_dim must be divisible by heads")
        if len(self.patch_size) != 2 or any(size < 1 for size in self.patch_size):
            raise ValueError("patch_size must contain two positive integers")
        if 4 % self.patch_size[0] or 9 % self.patch_size[1]:
            raise ValueError("patch_size must divide the 4x9 feature board")
        if not 0.0 <= self.dropout < 1.0 or not 0.0 <= self.drop_path < 1.0:
            raise ValueError("dropout and drop_path must be in [0, 1)")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["patch_size"] = list(self.patch_size)
        return result

    @classmethod
    def from_dict(cls, values: dict) -> "ViTConfig":
        values = dict(values)
        values["patch_size"] = tuple(values.get("patch_size", (1, 1)))
        return cls(**values)


VIT_PRESETS = {
    "vit_small": ViTConfig(
        name="vit_small", embed_dim=256, depth=6, heads=8, dropout=0.05, drop_path=0.05
    ),
    "vit_base": ViTConfig(
        name="vit_base", embed_dim=384, depth=8, heads=8, dropout=0.1, drop_path=0.1
    ),
    # Shape-compatible with advanced_training.create_large_multitask_vit.
    "vit_large": ViTConfig(
        name="vit_large", embed_dim=512, depth=8, heads=8, dropout=0.1, drop_path=0.1
    ),
}


def vit_config(name: str, **overrides) -> ViTConfig:
    if name not in VIT_PRESETS:
        raise ValueError(f"unknown ViT preset: {name}")
    return replace(VIT_PRESETS[name], **overrides)

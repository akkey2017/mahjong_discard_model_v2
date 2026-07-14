"""ViT-only snapshot model with fixed multi-task heads."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from mahjong_ai.data.multitask import TASK_CLASS_COUNTS

from .config import ViTConfig


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return inputs
        keep = 1.0 - self.drop_prob
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        return inputs * inputs.new_empty(shape).bernoulli_(keep).div_(keep)


class Attention(nn.Module):
    """Key-compatible attention implementation used by the current ViT."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.attn_dropout = dropout
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = inputs.shape
        qkv = self.to_qkv(inputs).reshape(batch, tokens, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(batch, tokens, dim)
        return self.to_out(output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        drop_path: float,
        mlp_ratio: float,
    ):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.drop_path(self.attn(self.norm1(inputs)))
        return inputs + self.drop_path(self.ffn(self.norm2(inputs)))


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: tuple[int, int]):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(inputs).flatten(2).transpose(1, 2)


class SnapshotVisionTransformer(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(
            config.in_channels, config.embed_dim, config.patch_size
        )
        self.use_cls_token = config.use_cls_token
        patches = (4 // config.patch_size[0]) * (9 // config.patch_size[1])
        tokens = patches + int(config.use_cls_token)
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.register_parameter("cls_token", None)
        self.pos_embed = nn.Parameter(torch.zeros(1, tokens, config.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(config.dropout)
        rates = [
            config.drop_path * index / max(config.depth - 1, 1)
            for index in range(config.depth)
        ]
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                config.embed_dim,
                config.heads,
                config.dropout,
                rates[index],
                config.mlp_ratio,
            )
            for index in range(config.depth)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)
        self.final_channels = config.embed_dim
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.patch_embed(inputs)
        if self.use_cls_token:
            cls = self.cls_token.expand(features.shape[0], -1, -1)
            features = torch.cat((cls, features), dim=1)
        features = self.dropout(features + self.pos_embed)
        for block in self.transformer_blocks:
            features = block(features)
        features = self.norm(features)
        return features[:, 0] if self.use_cls_token else features.mean(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_features(inputs)


class MahjongViTMultiTask(nn.Module):
    """One Snapshot ViT backbone followed by every Phase-5 task head."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.backbone = SnapshotVisionTransformer(config)
        # Names intentionally match legacy MultiTaskDiscardModel checkpoints.
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.heads = nn.ModuleDict({
            task: nn.Linear(config.embed_dim, classes)
            for task, classes in TASK_CLASS_COUNTS.items()
        })
        self.head_specs = dict(TASK_CLASS_COUNTS)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.dropout(self.norm(self.backbone.forward_features(inputs)))
        return {task: head(features) for task, head in self.heads.items()}


def create_vit(config: ViTConfig) -> MahjongViTMultiTask:
    return MahjongViTMultiTask(config)

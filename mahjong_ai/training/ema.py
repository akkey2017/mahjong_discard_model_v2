"""Exponential moving average independent of torch.compile wrappers."""

from __future__ import annotations

import copy

import torch

from .checkpoint import checkpoint_module


class ModelEMA:
    def __init__(self, model, decay: float):
        self.decay = float(decay)
        self.model = copy.deepcopy(checkpoint_module(model)).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        source = checkpoint_module(model)
        parameters = dict(source.named_parameters())
        for name, value in self.model.named_parameters():
            value.mul_(self.decay).add_(parameters[name].detach(), alpha=1.0 - self.decay)
        buffers = dict(source.named_buffers())
        for name, value in self.model.named_buffers():
            value.copy_(buffers[name].detach())

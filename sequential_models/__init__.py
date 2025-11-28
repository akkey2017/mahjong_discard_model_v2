"""
Sequential Models for Mahjong AI

This package provides models that learn the flow of an entire mahjong game (kyoku)
by processing sequences of game states and actions, rather than making single-step
predictions at isolated game states.

Key Features:
- LSTM-based sequential model
- Transformer Decoder-based sequential model  
- Sequence dataset that groups actions by kyoku (round)
- Training utilities for sequential models

Architecture Comparison:
- Original models: Predict discard at a single game state (classification)
- Sequential models: Process entire game sequences to learn temporal patterns

Usage:
    from sequential_models import (
        MahjongSequenceDataset,
        LSTMDiscardModel,
        TransformerDiscardModel,
        create_lstm_model,
        create_transformer_model
    )
"""

from .sequence_dataset import MahjongSequenceDataset, create_sequence_dataloaders
from .sequence_models import (
    LSTMDiscardModel,
    TransformerDiscardModel,
    create_lstm_model,
    create_transformer_model,
)

__all__ = [
    'MahjongSequenceDataset',
    'create_sequence_dataloaders',
    'LSTMDiscardModel',
    'TransformerDiscardModel',
    'create_lstm_model',
    'create_transformer_model',
]

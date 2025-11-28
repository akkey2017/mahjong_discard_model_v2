"""
Sequential Models for Mahjong Discard Prediction

This module provides sequence-based neural network architectures that process
entire game sequences rather than single states. These models can learn
temporal patterns and strategies that span multiple turns.

Key models:
- LSTMDiscardModel: Uses LSTM to process game state sequences
- TransformerDiscardModel: Uses Transformer Decoder for autoregressive prediction

Both models take sequences of game states and predict the discard action
at each step, enabling learning of game flow patterns.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateEncoder(nn.Module):
    """
    Encode a single game state (380, 4, 9) into a fixed-size representation.
    
    This is used as the first stage in sequential models to compress
    each game state before feeding into the sequence model.
    """
    
    def __init__(self, in_channels=380, hidden_dim=256, dropout=0.1):
        super().__init__()
        
        # Convolutional encoder
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, hidden_dim, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        
        # Global pooling and projection
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        """
        Encode game state(s).
        
        Args:
            x: (batch, channels, H, W) or (batch, seq_len, channels, H, W)
        
        Returns:
            (batch, hidden_dim) or (batch, seq_len, hidden_dim)
        """
        has_seq_dim = x.dim() == 5
        
        if has_seq_dim:
            batch, seq_len, C, H, W = x.shape
            x = x.view(batch * seq_len, C, H, W)
        
        # Convolutional layers
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        
        # Pool and flatten
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        if has_seq_dim:
            x = x.view(batch, seq_len, -1)
        
        return x


class LSTMDiscardModel(nn.Module):
    """
    LSTM-based sequential discard prediction model.
    
    Architecture:
    1. StateEncoder: Compress each game state (380, 4, 9) -> (hidden_dim,)
    2. LSTM: Process sequence of encoded states
    3. Output layer: Predict discard action at each step
    
    This model can learn temporal patterns across multiple turns in a game.
    """
    
    def __init__(self, state_encoder, hidden_dim=256, num_layers=2, 
                 num_classes=34, dropout=0.1, bidirectional=False):
        """
        Args:
            state_encoder: StateEncoder module for encoding game states
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            num_classes: Number of output classes (34 for mahjong tiles)
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.state_encoder = state_encoder
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=state_encoder.hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output projection
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, lengths=None):
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, 380, 4, 9) - sequence of game states
            lengths: (batch,) - actual sequence lengths (for packed sequences)
        
        Returns:
            (batch, seq_len, num_classes) - logits for each position
        """
        batch_size, seq_len = x.shape[:2]
        
        # Encode all states
        encoded = self.state_encoder(x)  # (batch, seq_len, hidden_dim)
        
        # Process with LSTM
        if lengths is not None:
            # Pack sequence for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                encoded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=seq_len
            )
        else:
            lstm_out, _ = self.lstm(encoded)
        
        # Output projection
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)  # (batch, seq_len, num_classes)
        
        return logits
    
    def predict_step(self, state, hidden=None):
        """
        Predict a single step (for inference/interactive use).
        
        Args:
            state: (batch, 380, 4, 9) - single game state
            hidden: LSTM hidden state from previous step
        
        Returns:
            Tuple of (logits, new_hidden)
        """
        # Encode state
        encoded = self.state_encoder(state)  # (batch, hidden_dim)
        encoded = encoded.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # LSTM step
        lstm_out, hidden = self.lstm(encoded, hidden)
        
        # Output
        lstm_out = self.dropout(lstm_out.squeeze(1))
        logits = self.fc(lstm_out)
        
        return logits, hidden


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Handle both even and odd d_model
        pe[:, 0::2] = torch.sin(position * div_term)
        # For odd d_model, cos dimension is one less than sin
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerDiscardModel(nn.Module):
    """
    Transformer-based sequential discard prediction model.
    
    Architecture:
    1. StateEncoder: Compress each game state (380, 4, 9) -> (d_model,)
    2. Positional Encoding: Add position information
    3. Transformer Decoder: Process sequence with self-attention
    4. Output layer: Predict discard action at each step
    
    Uses causal masking to ensure the model only attends to previous positions,
    enabling autoregressive prediction during inference.
    """
    
    def __init__(self, state_encoder, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=1024, num_classes=34, dropout=0.1, max_seq_len=50):
        """
        Args:
            state_encoder: StateEncoder module for encoding game states
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            num_classes: Number of output classes (34 for mahjong tiles)
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.state_encoder = state_encoder
        self.d_model = d_model
        
        # Project state encoder output to d_model if different
        if state_encoder.hidden_dim != d_model:
            self.input_proj = nn.Linear(state_encoder.hidden_dim, d_model)
        else:
            self.input_proj = nn.Identity()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def _generate_causal_mask(self, seq_len, device):
        """Generate causal mask for autoregressive attention."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        return mask
    
    def _generate_padding_mask(self, lengths, max_len, device):
        """Generate padding mask from sequence lengths."""
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=device).expand(batch_size, max_len) >= lengths.unsqueeze(1)
        return mask  # Returns bool tensor
    
    def forward(self, x, lengths=None):
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, 380, 4, 9) - sequence of game states
            lengths: (batch,) - actual sequence lengths
        
        Returns:
            (batch, seq_len, num_classes) - logits for each position
        """
        batch_size, seq_len = x.shape[:2]
        device = x.device
        
        # Encode all states
        encoded = self.state_encoder(x)  # (batch, seq_len, hidden_dim)
        encoded = self.input_proj(encoded)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        encoded = self.pos_encoder(encoded)
        
        # Generate masks
        causal_mask = self._generate_causal_mask(seq_len, device)
        
        if lengths is not None:
            padding_mask = self._generate_padding_mask(lengths, seq_len, device)
        else:
            padding_mask = None
        
        # Transformer decoder (using encoded as both memory and target for decoder-only setup)
        # For a decoder-only architecture, we use self-attention with causal masking
        output = self.transformer_decoder(
            encoded,
            encoded,  # Use same sequence as memory
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask
        )
        
        # Output projection
        output = self.dropout(output)
        logits = self.fc(output)  # (batch, seq_len, num_classes)
        
        return logits
    
    def predict_step(self, states, past_states=None):
        """
        Predict using all past states (for inference).
        
        For Transformer models, we need all past states to make predictions
        (unlike LSTM which has a hidden state).
        
        Args:
            states: (batch, seq_len, 380, 4, 9) - all states up to current
            past_states: Not used (kept for API compatibility with LSTM)
        
        Returns:
            logits for the last position: (batch, num_classes)
        """
        logits = self.forward(states)
        return logits[:, -1, :]  # Return only the last position


# ==================== Factory Functions ====================

def create_lstm_model(in_channels=380, hidden_dim=256, num_layers=2,
                      num_classes=34, dropout=0.1, bidirectional=False):
    """
    Create an LSTM-based discard model.
    
    Args:
        in_channels: Number of input channels (380 for state tensor)
        hidden_dim: Hidden dimension for both encoder and LSTM
        num_layers: Number of LSTM layers
        num_classes: Number of output classes (34 tiles)
        dropout: Dropout rate
        bidirectional: Whether to use bidirectional LSTM
    
    Returns:
        LSTMDiscardModel instance
    """
    state_encoder = StateEncoder(in_channels, hidden_dim, dropout)
    model = LSTMDiscardModel(
        state_encoder=state_encoder,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional
    )
    return model


def create_transformer_model(in_channels=380, d_model=256, nhead=8,
                             num_layers=4, dim_feedforward=1024,
                             num_classes=34, dropout=0.1, max_seq_len=50):
    """
    Create a Transformer-based discard model.
    
    Args:
        in_channels: Number of input channels (380 for state tensor)
        d_model: Transformer model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Dimension of feedforward network
        num_classes: Number of output classes (34 tiles)
        dropout: Dropout rate
        max_seq_len: Maximum sequence length
    
    Returns:
        TransformerDiscardModel instance
    """
    state_encoder = StateEncoder(in_channels, d_model, dropout)
    model = TransformerDiscardModel(
        state_encoder=state_encoder,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        num_classes=num_classes,
        dropout=dropout,
        max_seq_len=max_seq_len
    )
    return model

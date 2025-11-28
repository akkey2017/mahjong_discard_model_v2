"""
Sequence Dataset for Mahjong Game Records

This module provides a dataset implementation that groups game actions by kyoku (round)
to enable learning the sequential flow of a game, rather than treating each action
as an independent sample.

Key differences from the original dataset:
- Groups samples by kyoku_id to maintain game sequence
- Returns sequences of (state, action) pairs for a single player in a round
- Supports variable-length sequences with padding and masking
"""

import json
import os
import sys
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# Import from parent module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mahjong_ai_features import StateEncoderV2, _process_single_number, FEATURE_TILE_MAP


class MahjongSequenceDataset(Dataset):
    """
    Dataset for sequential mahjong game records.
    
    Groups actions by (game_file, kyoku_index, player_id) to create sequences
    representing a single player's actions throughout one round (kyoku).
    
    Each sample is a sequence of:
    - state tensors: (seq_len, 380, 4, 9) - game states before each action
    - action labels: (seq_len,) - tile IDs (0-33) that were discarded
    - sequence length: int - actual length before padding
    
    This enables models to learn temporal patterns in a player's decision-making
    throughout a round, rather than treating each decision independently.
    """
    
    def __init__(self, zip_path=None, max_files=10000, sequences=None, 
                 max_seq_len=30, verbose=True):
        """
        Initialize the dataset.
        
        Args:
            zip_path: Path to ZIP file containing game logs
            max_files: Maximum number of files to load from the ZIP
            sequences: Pre-loaded sequences (for creating subsets)
            max_seq_len: Maximum sequence length (longer sequences are truncated)
            verbose: Whether to show progress during loading
        """
        self.max_seq_len = max_seq_len
        
        if sequences is not None:
            self.sequences = sequences
            return
        
        self.sequences = []
        
        if zip_path is None:
            return
        
        # Temporary storage: {(file_idx, kyoku_idx, player_id): [(log_idx, label), ...]}
        kyoku_actions = {}
        # Store kyoku_log for each kyoku
        kyoku_logs = {}
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                txts = [n for n in zf.namelist() if n.endswith('.txt')]
                
                iterator = txts[:max_files]
                if verbose:
                    iterator = tqdm(iterator, desc="Loading game records",
                                   file=sys.stderr, dynamic_ncols=True, mininterval=0.1)
                
                for file_idx, name in enumerate(iterator):
                    try:
                        raw = zf.read(name).decode("utf-8")
                        game_data = json.loads(raw)
                        
                        if 'log' not in game_data:
                            continue
                        
                        for kyoku_idx, kyoku_log in enumerate(game_data['log']):
                            # Skip incomplete rounds
                            if not kyoku_log or 'qipai' not in kyoku_log[0]:
                                continue
                            
                            # Store the kyoku_log for later use
                            for log_idx, move in enumerate(kyoku_log):
                                if 'dapai' in move:
                                    p_id = move['dapai']['l']
                                    tile_str = move['dapai']['p'].replace('*', '').replace('_', '')
                                    
                                    if tile_str in FEATURE_TILE_MAP:
                                        tile_id_37 = FEATURE_TILE_MAP[tile_str]
                                        label = _process_single_number(tile_id_37)
                                        
                                        # Create key for this player's sequence in this round
                                        key = (file_idx, kyoku_idx, p_id)
                                        
                                        if key not in kyoku_actions:
                                            kyoku_actions[key] = []
                                            kyoku_logs[key] = kyoku_log
                                        
                                        kyoku_actions[key].append((log_idx, label))
                    
                    except json.JSONDecodeError:
                        if verbose:
                            print(f"Warning: Failed to parse JSON in {name}")
                        continue
                    except Exception as e:
                        if verbose:
                            print(f"Warning: Error processing {name}: {e}")
                        continue
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {zip_path}")
        except zipfile.BadZipFile:
            raise ValueError(f"Invalid ZIP file: {zip_path}")
        
        # Convert to list of sequences
        for key, actions in kyoku_actions.items():
            if len(actions) > 0:  # Only keep non-empty sequences
                file_idx, kyoku_idx, p_id = key
                kyoku_log = kyoku_logs[key]
                self.sequences.append({
                    'kyoku_log': kyoku_log,
                    'player_id': p_id,
                    'actions': actions  # List of (log_idx, label) tuples
                })
        
        if verbose:
            print(f"Created {len(self.sequences)} sequences from {len(kyoku_actions)} player-rounds")
    
    def __len__(self):
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a single sequence from the dataset.
        
        Args:
            idx: Index of the sequence
        
        Returns:
            Tuple of (state_tensors, labels, seq_length)
            - state_tensors: (max_seq_len, 380, 4, 9) tensor
            - labels: (max_seq_len,) tensor of action labels
            - seq_length: actual sequence length
        """
        seq_data = self.sequences[idx]
        kyoku_log = seq_data['kyoku_log']
        p_id = seq_data['player_id']
        actions = seq_data['actions']
        
        # Truncate if needed
        actions = actions[:self.max_seq_len]
        seq_len = len(actions)
        
        # Encode each state in the sequence
        encoder = StateEncoderV2(kyoku_log, p_id)
        
        state_tensors = []
        labels = []
        
        for log_idx, label in actions:
            state_tensor = encoder.encode(log_idx)
            state_tensors.append(state_tensor)
            labels.append(label)
        
        # Pad to max_seq_len
        while len(state_tensors) < self.max_seq_len:
            # Use zero tensor for padding
            state_tensors.append(torch.zeros(380, 4, 9))
            labels.append(-100)  # Ignore index for cross-entropy loss
        
        # Stack tensors
        state_tensors = torch.stack(state_tensors)  # (max_seq_len, 380, 4, 9)
        labels = torch.tensor(labels, dtype=torch.long)  # (max_seq_len,)
        
        return state_tensors, labels, seq_len
    
    def get_statistics(self):
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if not self.sequences:
            return {"total_sequences": 0}
        
        seq_lengths = [len(seq['actions']) for seq in self.sequences]
        total_actions = sum(seq_lengths)
        
        return {
            "total_sequences": len(self.sequences),
            "total_actions": total_actions,
            "avg_seq_length": total_actions / len(self.sequences) if self.sequences else 0,
            "min_seq_length": min(seq_lengths) if seq_lengths else 0,
            "max_seq_length": max(seq_lengths) if seq_lengths else 0,
        }


def collate_sequences(batch):
    """
    Collate function for sequence batches.
    
    Args:
        batch: List of (state_tensors, labels, seq_length) tuples
    
    Returns:
        Tuple of (batched_states, batched_labels, lengths)
    """
    states, labels, lengths = zip(*batch)
    
    # Stack into batch tensors
    states = torch.stack(states)  # (batch, max_seq_len, 380, 4, 9)
    labels = torch.stack(labels)  # (batch, max_seq_len)
    lengths = torch.tensor(lengths, dtype=torch.long)  # (batch,)
    
    return states, labels, lengths


def create_sequence_dataloaders(dataset, train_ratio=0.9, batch_size=32, 
                                num_workers=2, pin_memory=True, seed=42):
    """
    Create train and validation data loaders for sequence dataset.
    
    Args:
        dataset: MahjongSequenceDataset instance
        train_ratio: Ratio of data to use for training
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        seed: Random seed for reproducible splits
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    
    # Split dataset
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_sequences
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_sequences
    )
    
    return train_loader, val_loader

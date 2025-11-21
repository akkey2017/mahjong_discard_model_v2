"""
Dataset handling for mahjong game records.
"""

import json
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from mahjong_ai_features import StateEncoderV2, _process_single_number, FEATURE_TILE_MAP


class MahjongDataset(Dataset):
    """
    Dataset for mahjong game records.
    
    Loads game data from a ZIP file containing JSON game logs and converts
    them into training samples.
    """
    
    def __init__(self, zip_path=None, max_files=10000, samples=None, verbose=True):
        """
        Initialize the dataset.
        
        Args:
            zip_path: Path to ZIP file containing game logs (*.txt with JSON)
            max_files: Maximum number of files to load from the ZIP
            samples: Pre-loaded samples (for creating subsets)
            verbose: Whether to show progress during loading
        """
        if samples is not None:
            self.samples = samples
            return
        
        self.samples = []
        
        if zip_path is None:
            return
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                txts = [n for n in zf.namelist() if n.endswith('.txt')]
                
                iterator = txts[:max_files]
                if verbose:
                    iterator = tqdm(iterator, desc="Loading game records")
                
                for name in iterator:
                    try:
                        raw = zf.read(name).decode("utf-8")
                        game_data = json.loads(raw)
                        
                        if 'log' not in game_data:
                            continue
                        
                        for kyoku_log in game_data['log']:
                            # Skip incomplete rounds
                            if not kyoku_log or 'qipai' not in kyoku_log[0]:
                                continue
                            
                            for i, move in enumerate(kyoku_log):
                                if 'dapai' in move:
                                    p_id = move['dapai']['l']
                                    tile_str = move['dapai']['p'].replace('*', '').replace('_', '')
                                    
                                    if tile_str in FEATURE_TILE_MAP:
                                        tile_id_37 = FEATURE_TILE_MAP[tile_str]
                                        label = _process_single_number(tile_id_37)
                                        self.samples.append((kyoku_log, i, p_id, 'discard', label))
                    
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
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Tuple of (state_tensor, label, action_type)
        """
        kyoku_log, log_idx, p_id, action_type, label = self.samples[idx]
        
        # Encode the game state
        encoder = StateEncoderV2(kyoku_log, p_id)
        state_tensor = encoder.encode(log_idx)
        
        return state_tensor, label, action_type
    
    def filter_by_action(self, action_type):
        """
        Create a new dataset containing only samples of a specific action type.
        
        Args:
            action_type: Action type to filter by (e.g., 'discard')
        
        Returns:
            New MahjongDataset with filtered samples
        """
        filtered_samples = [s for s in self.samples if s[3] == action_type]
        return MahjongDataset(samples=filtered_samples, verbose=False)
    
    def get_statistics(self):
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if not self.samples:
            return {"total": 0}
        
        action_counts = {}
        for sample in self.samples:
            action = sample[3]
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            "total": len(self.samples),
            "action_counts": action_counts
        }


def create_dataloaders(dataset, train_ratio=0.9, batch_size=64, num_workers=2, 
                       pin_memory=True, seed=42):
    """
    Create train and validation data loaders from a dataset.
    
    Args:
        dataset: MahjongDataset instance
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
    
    # Create data loaders
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader

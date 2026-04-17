"""
Dataset helpers for training with multiple ZIP archives.
"""

from pathlib import Path
import sys

# Ensure repository root is importable when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import MahjongDataset  # noqa: E402


class MultiZipMahjongDataset(MahjongDataset):
    """
    Aggregated dataset that loads samples from multiple ZIP archives.

    This class reuses the base :class:`MahjongDataset` logic while allowing
    multiple data archives to be specified. It keeps a simple breakdown of
    how many samples were loaded from each source for reporting.
    """

    def __init__(self, zip_paths, max_files_per_zip=10000, verbose=True,
                 collect_all_actions=False, include_fulou_negatives=False):
        if isinstance(zip_paths, (str, Path)):
            zip_paths = [zip_paths]

        resolved_paths = [Path(p).expanduser() for p in zip_paths or []]
        if not resolved_paths:
            raise ValueError("At least one ZIP path must be provided")

        combined_samples = []
        combined_game_ids = []
        self.source_counts = {}

        for path in resolved_paths:
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {path}")

            dataset = MahjongDataset(
                zip_path=str(path),
                max_files=max_files_per_zip,
                verbose=verbose,
                collect_all_actions=collect_all_actions,
                include_fulou_negatives=include_fulou_negatives,
            )
            combined_samples.extend(dataset.samples)
            # Prefix game ids with source archive so they're unique across zips
            combined_game_ids.extend(
                f"{path.name}::{gid}" for gid in dataset.game_ids
            )
            self.source_counts[str(path)] = len(dataset)

        # Initialize parent with the aggregated samples
        super().__init__(samples=combined_samples, verbose=False)
        self.game_ids = combined_game_ids

    def get_statistics(self):
        stats = super().get_statistics()
        stats["source_counts"] = self.source_counts
        return stats

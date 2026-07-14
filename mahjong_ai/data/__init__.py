"""Compact dataset preparation and shard format."""

from .normalize import NormalizedGame, NormalizedRound, ValidationError, normalize_game
from .schema import DATASET_SCHEMA_VERSION, EVENT_DTYPE, METADATA_DTYPE, ROUND_DTYPE
from .shard_writer import verify_dataset, verify_shard
from .multitask import (
    TARGET_SCHEMA_VERSION,
    NEGATIVE_SAMPLING_TASKS,
    TASK_CLASS_COUNTS,
    TASK_INDEX,
    TASK_NAMES,
    MultiTaskMetrics,
    MultiTaskSample,
    MultiTaskSampleBuilder,
    MultiTaskTarget,
    NegativeSamplingConfig,
    StreamingMultiTaskDataset,
    TaskSamplingPolicy,
    build_multitask_dataloader,
    masked_multitask_loss,
    unified_multitask_collate,
)
from .streaming_dataset import (
    CompactRoundRecord,
    CompactShardReader,
    StreamingRoundDataset,
    TensorRoundRecord,
    build_streaming_dataloader,
)

__all__ = [
    "DATASET_SCHEMA_VERSION",
    "EVENT_DTYPE",
    "METADATA_DTYPE",
    "ROUND_DTYPE",
    "NormalizedGame",
    "NormalizedRound",
    "ValidationError",
    "normalize_game",
    "verify_dataset",
    "verify_shard",
    "CompactRoundRecord",
    "CompactShardReader",
    "StreamingRoundDataset",
    "TensorRoundRecord",
    "build_streaming_dataloader",
    "TARGET_SCHEMA_VERSION",
    "NEGATIVE_SAMPLING_TASKS",
    "TASK_CLASS_COUNTS",
    "TASK_INDEX",
    "TASK_NAMES",
    "MultiTaskMetrics",
    "MultiTaskSample",
    "MultiTaskSampleBuilder",
    "MultiTaskTarget",
    "NegativeSamplingConfig",
    "StreamingMultiTaskDataset",
    "TaskSamplingPolicy",
    "build_multitask_dataloader",
    "masked_multitask_loss",
    "unified_multitask_collate",
]

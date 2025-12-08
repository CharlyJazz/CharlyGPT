"""
Streaming Dataset Module for GPT Pre-training

This module provides memory-efficient streaming data loading for training
large language models on datasets that don't fit in RAM.

Main components:
- StreamingGPTDataset: IterableDataset that streams from HuggingFace
- create_streaming_dataloaders: Factory function to create train/val loaders
- DatasetCheckpointManager: Handles saving/loading dataset state for resuming

Usage:
    from dataset import create_streaming_dataloaders, DatasetCheckpointManager
    
    train_loader, val_loader = create_streaming_dataloaders(
        batch_size=8,
        max_length=256,
        num_samples=1000000
    )
    
    checkpoint_manager = DatasetCheckpointManager(train_loader)
    state = checkpoint_manager.get_state()
    checkpoint_manager.restore_state(state)
"""

from .streaming_dataset import (
    StreamingGPTDataset,
    StreamingGPTDatasetWithSkip,
    DatasetState,
)

from .dataloader_factory import (
    create_streaming_dataloaders,
    create_single_streaming_dataloader,
    create_dataloaders_from_config,
    get_dataloader_info,
    print_dataloader_status,
    HAS_STATEFUL_DATALOADER,
)

from .checkpoint_manager import (
    DatasetCheckpointManager,
    CheckpointState,
)

from .tokenizer_utils import (
    get_tokenizer,
    text_to_token_ids,
    token_ids_to_text,
    get_eos_token_id,
    get_vocab_size,
    count_tokens,
)

from .config import (
    StreamingConfig,
    DEFAULT_CONFIG,
    PRESETS,
    get_preset,
)

__all__ = [
    # Dataset classes
    "StreamingGPTDataset",
    "StreamingGPTDatasetWithSkip",
    "DatasetState",
    # Factory functions
    "create_streaming_dataloaders",
    "create_single_streaming_dataloader",
    "create_dataloaders_from_config",
    "get_dataloader_info",
    "print_dataloader_status",
    "HAS_STATEFUL_DATALOADER",
    # Checkpoint management
    "DatasetCheckpointManager",
    "CheckpointState",
    # Tokenizer utilities
    "get_tokenizer",
    "text_to_token_ids", 
    "token_ids_to_text",
    "get_eos_token_id",
    "get_vocab_size",
    "count_tokens",
    # Config
    "StreamingConfig",
    "DEFAULT_CONFIG",
    "PRESETS",
    "get_preset",
]

__version__ = "1.0.0"

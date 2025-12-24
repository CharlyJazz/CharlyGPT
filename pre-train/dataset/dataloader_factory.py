"""
DataLoader Factory for Streaming Datasets

Provides factory functions to create train/validation dataloaders
with optional StatefulDataLoader support for mid-epoch checkpointing.
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Union

from .streaming_dataset import StreamingGPTDataset
from .config import StreamingConfig, DEFAULT_CONFIG


# Try to import StatefulDataLoader for mid-epoch checkpointing
try:
    from torchdata.stateful_dataloader import StatefulDataLoader
    HAS_STATEFUL_DATALOADER = True
except ImportError:
    # Fallback to regular DataLoader
    StatefulDataLoader = DataLoader
    HAS_STATEFUL_DATALOADER = False


def create_streaming_dataloaders(
    batch_size: int = 8,
    max_length: int = 256,
    num_samples: Optional[int] = None,
    buffer_size: int = 10000,
    seed: int = 42,
    train_ratio: float = 0.9,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    dataset_name: str = "PleIAs/SYNTH",
    language_filter: Optional[str] = "en",
    use_stateful: bool = True,
    local_path: Optional[str] = None,
) -> Tuple[Union[DataLoader, "StatefulDataLoader"], Union[DataLoader, "StatefulDataLoader"]]:
    """
    Create train and validation streaming dataloaders.
    
    Args:
        batch_size: Batch size for training
        max_length: Sequence length (context size)
        num_samples: Max raw samples to process (None = unlimited)
        buffer_size: Shuffle buffer size for approximate shuffling
        seed: Random seed for reproducibility
        train_ratio: Ratio of data for training (rest for validation)
        num_workers: DataLoader workers (0 recommended for streaming)
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        dataset_name: HuggingFace dataset name
        language_filter: Filter by language (None = no filter)
        use_stateful: Use StatefulDataLoader if available (for checkpointing)
        local_path: Local path to dataset (None = use HuggingFace Hub)
    
    Returns:
        Tuple of (train_loader, val_loader)
        
    Example:
        >>> train_loader, val_loader = create_streaming_dataloaders(
        ...     batch_size=8,
        ...     max_length=256,
        ...     num_samples=100000
        ... )
        >>> for batch in train_loader:
        ...     input_ids, target_ids = batch
        ...     print(input_ids.shape)  # torch.Size([8, 256])
        ...     break
    """
    # Create train dataset
    train_dataset = StreamingGPTDataset(
        max_length=max_length,
        num_samples=num_samples,
        buffer_size=buffer_size,
        seed=seed,
        split="train",
        train_ratio=train_ratio,
        dataset_name=dataset_name,
        language_filter=language_filter,
        local_path=local_path,
    )
    
    # Create validation dataset
    val_dataset = StreamingGPTDataset(
        max_length=max_length,
        num_samples=num_samples,
        buffer_size=buffer_size,
        seed=seed,
        split="val",
        train_ratio=train_ratio,
        dataset_name=dataset_name,
        language_filter=language_filter,
        local_path=local_path,
    )
    
    # Select DataLoader class
    LoaderClass = StatefulDataLoader if (use_stateful and HAS_STATEFUL_DATALOADER) else DataLoader
    
    # Create dataloaders
    train_loader = LoaderClass(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    
    val_loader = LoaderClass(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # Don't drop last for validation
    )
    
    return train_loader, val_loader


def create_single_streaming_dataloader(
    split: str = "train",
    batch_size: int = 8,
    max_length: int = 256,
    num_samples: Optional[int] = None,
    buffer_size: int = 10000,
    seed: int = 42,
    train_ratio: float = 0.9,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    dataset_name: str = "PleIAs/SYNTH",
    language_filter: Optional[str] = "en",
    use_stateful: bool = True,
) -> Union[DataLoader, "StatefulDataLoader"]:
    """
    Create a single streaming dataloader for either train or validation.
    
    Args:
        split: Either 'train' or 'val'
        ... (same as create_streaming_dataloaders)
    
    Returns:
        DataLoader or StatefulDataLoader instance
    """
    dataset = StreamingGPTDataset(
        max_length=max_length,
        num_samples=num_samples,
        buffer_size=buffer_size,
        seed=seed,
        split=split,
        train_ratio=train_ratio,
        dataset_name=dataset_name,
        language_filter=language_filter,
    )
    
    LoaderClass = StatefulDataLoader if (use_stateful and HAS_STATEFUL_DATALOADER) else DataLoader
    
    return LoaderClass(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last if split == "train" else False,
    )


def create_dataloaders_from_config(
    config: StreamingConfig,
    use_stateful: bool = True,
) -> Tuple[Union[DataLoader, "StatefulDataLoader"], Union[DataLoader, "StatefulDataLoader"]]:
    """
    Create dataloaders from a StreamingConfig object.
    
    Args:
        config: StreamingConfig with all parameters
        use_stateful: Use StatefulDataLoader if available
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    return create_streaming_dataloaders(
        batch_size=config.batch_size,
        max_length=config.max_length,
        num_samples=config.num_samples,
        buffer_size=config.buffer_size,
        seed=config.seed,
        train_ratio=config.train_ratio,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        dataset_name=config.dataset_name,
        language_filter=config.language_filter,
        use_stateful=use_stateful,
    )


def get_dataloader_info(dataloader: DataLoader) -> dict:
    """
    Get information about a dataloader.
    
    Args:
        dataloader: DataLoader instance
    
    Returns:
        Dictionary with dataloader info
    """
    dataset = dataloader.dataset
    
    info = {
        "batch_size": dataloader.batch_size,
        "num_workers": dataloader.num_workers,
        "pin_memory": dataloader.pin_memory,
        "drop_last": dataloader.drop_last,
        "is_stateful": HAS_STATEFUL_DATALOADER and isinstance(dataloader, StatefulDataLoader),
    }
    
    # Add dataset info if available
    if hasattr(dataset, "max_length"):
        info["max_length"] = dataset.max_length
    if hasattr(dataset, "buffer_size"):
        info["buffer_size"] = dataset.buffer_size
    if hasattr(dataset, "num_samples"):
        info["num_samples"] = dataset.num_samples
    if hasattr(dataset, "split"):
        info["split"] = dataset.split
    
    return info


def check_stateful_dataloader_available() -> bool:
    """
    Check if StatefulDataLoader is available.
    
    Returns:
        True if torchdata.stateful_dataloader is installed
    """
    return HAS_STATEFUL_DATALOADER


def print_dataloader_status():
    """Print status of dataloader capabilities"""
    print("=" * 50)
    print("DataLoader Status")
    print("=" * 50)
    print(f"StatefulDataLoader available: {HAS_STATEFUL_DATALOADER}")
    
    if not HAS_STATEFUL_DATALOADER:
        print("\n[WARN] To enable mid-epoch checkpointing, install torchdata:")
        print("   pip install torchdata>=0.8.0")
    else:
        print("\n[OK] Mid-epoch checkpointing is available")
    
    print("=" * 50)

"""
Configuration classes for Streaming Dataset

Provides dataclass-based configuration for all streaming dataset parameters.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StreamingConfig:
    """
    Configuration for streaming GPT dataset.
    
    Attributes:
        max_length: Sequence length for training (context size)
        num_samples: Maximum number of raw samples to process (None = unlimited)
        buffer_size: Shuffle buffer size (higher = better randomization, more RAM)
        seed: Random seed for reproducibility
        train_ratio: Ratio of data for training (rest goes to validation)
        dataset_name: HuggingFace dataset name
        language_filter: Filter by language (None = no filter)
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers (0 recommended for streaming)
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
    """
    # Sequence configuration
    max_length: int = 256
    
    # Dataset size
    num_samples: Optional[int] = None
    
    # Shuffle configuration
    buffer_size: int = 10000
    seed: int = 42
    
    # Train/val split
    train_ratio: float = 0.9
    
    # Data source
    dataset_name: str = "PleIAs/SYNTH"
    language_filter: Optional[str] = "en"
    
    # DataLoader configuration
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        assert 0 < self.max_length <= 8192, f"max_length must be in (0, 8192], got {self.max_length}"
        assert self.buffer_size > 0, f"buffer_size must be positive, got {self.buffer_size}"
        assert 0 < self.train_ratio < 1, f"train_ratio must be in (0, 1), got {self.train_ratio}"
        assert self.batch_size > 0, f"batch_size must be positive, got {self.batch_size}"
        assert self.num_workers >= 0, f"num_workers must be non-negative, got {self.num_workers}"
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            "max_length": self.max_length,
            "num_samples": self.num_samples,
            "buffer_size": self.buffer_size,
            "seed": self.seed,
            "train_ratio": self.train_ratio,
            "dataset_name": self.dataset_name,
            "language_filter": self.language_filter,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "StreamingConfig":
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


# Default configuration
DEFAULT_CONFIG = StreamingConfig()


# Preset configurations for common use cases
PRESETS = {
    "small_test": StreamingConfig(
        max_length=128,
        num_samples=1000,
        buffer_size=100,
        batch_size=4,
    ),
    "medium_experiment": StreamingConfig(
        max_length=256,
        num_samples=100000,
        buffer_size=5000,
        batch_size=8,
    ),
    "large_training": StreamingConfig(
        max_length=512,
        num_samples=None,  # Full dataset
        buffer_size=50000,
        batch_size=16,
    ),
    "debug": StreamingConfig(
        max_length=64,
        num_samples=100,
        buffer_size=50,
        batch_size=2,
    ),
}


def get_preset(name: str) -> StreamingConfig:
    """Get a preset configuration by name"""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESETS.keys())}")
    return PRESETS[name]

"""
Streaming GPT Dataset

Memory-efficient IterableDataset that streams data from HuggingFace,
tokenizes on-the-fly, and yields fixed-length sequences for language modeling.
"""

import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from typing import Optional, Iterator, Tuple, Dict, Any
from dataclasses import dataclass
import hashlib

from .tokenizer_utils import get_tokenizer, get_eos_token_id


@dataclass
class DatasetState:
    """State of the streaming dataset for checkpointing"""
    sequences_yielded: int
    samples_processed: int
    current_epoch: int
    seed: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequences_yielded": self.sequences_yielded,
            "samples_processed": self.samples_processed,
            "current_epoch": self.current_epoch,
            "seed": self.seed,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetState":
        return cls(
            sequences_yielded=d.get("sequences_yielded", 0),
            samples_processed=d.get("samples_processed", 0),
            current_epoch=d.get("current_epoch", 0),
            seed=d.get("seed", 42),
        )


class StreamingGPTDataset(IterableDataset):
    """
    Streaming dataset for GPT pre-training.
    
    Loads data from HuggingFace in streaming mode, tokenizes on-the-fly,
    and yields fixed-length sequences for language modeling.
    
    Memory usage: ~1-2 GB regardless of dataset size.
    
    Args:
        max_length: Sequence length for training (context size)
        num_samples: Maximum number of raw samples to process (None = unlimited)
        buffer_size: Shuffle buffer size (higher = better randomization, more RAM)
        seed: Random seed for reproducibility
        split: 'train' or 'val' - determines which portion of data to use
        train_ratio: Ratio of data for training (default 0.9)
        dataset_name: HuggingFace dataset name
        language_filter: Filter by language (default 'en', None = no filter)
        
    Example:
        >>> dataset = StreamingGPTDataset(max_length=256, num_samples=10000)
        >>> for input_ids, target_ids in dataset:
        ...     print(input_ids.shape)  # torch.Size([256])
        ...     break
    """
    
    def __init__(
        self,
        max_length: int = 256,
        num_samples: Optional[int] = None,
        buffer_size: int = 10000,
        seed: int = 42,
        split: str = "train",
        train_ratio: float = 0.9,
        dataset_name: str = "PleIAs/SYNTH",
        language_filter: Optional[str] = "en",
    ):
        super().__init__()
        
        # Validate arguments
        assert split in ("train", "val"), f"split must be 'train' or 'val', got '{split}'"
        assert 0 < train_ratio < 1, f"train_ratio must be in (0, 1), got {train_ratio}"
        assert max_length > 0, f"max_length must be positive, got {max_length}"
        assert buffer_size > 0, f"buffer_size must be positive, got {buffer_size}"
        
        self.max_length = max_length
        self.num_samples = num_samples
        self.buffer_size = buffer_size
        self.seed = seed
        self.split = split
        self.train_ratio = train_ratio
        self.dataset_name = dataset_name
        self.language_filter = language_filter
        
        # Initialize tokenizer
        self.tokenizer = get_tokenizer()
        self.eos_token_id = get_eos_token_id(self.tokenizer)
        
        # State tracking for checkpointing
        self._state = DatasetState(
            sequences_yielded=0,
            samples_processed=0,
            current_epoch=0,
            seed=seed,
        )
        
        # Skip count for resuming (set by load_state_dict)
        self._skip_sequences = 0
    
    def _format_sample(self, item: dict) -> Optional[str]:
        """
        Format a single sample into text.
        
        Override this method to customize formatting for different datasets.
        
        Args:
            item: Dictionary from HuggingFace dataset
            
        Returns:
            Formatted text string or None if sample should be skipped
        """
        query = item.get("query", "")
        reasoning = item.get("synthetic_reasoning", "")
        answer = item.get("synthetic_answer", "")
        
        if not query or not answer:
            return None
        
        if reasoning:
            return f"Q: {query}\n\nReasoning:\n{reasoning}\n\nA: {answer}"
        else:
            return f"Q: {query}\n\nA: {answer}"
    
    def _should_include_sample(self, sample_idx: int) -> bool:
        """
        Determine if sample belongs to train or val split.
        
        Uses deterministic hashing for consistent splitting across runs.
        """
        # Hash the sample index for deterministic splitting
        hash_val = int(hashlib.md5(str(sample_idx).encode()).hexdigest(), 16)
        is_train = (hash_val % 100) < (self.train_ratio * 100)
        
        if self.split == "train":
            return is_train
        else:
            return not is_train
    
    def _create_hf_dataset(self):
        """Create the HuggingFace streaming dataset"""
        dataset = load_dataset(
            self.dataset_name,
            split="train",
            streaming=True,
        )
        
        # Effective seed includes epoch for different shuffling each epoch
        effective_seed = self.seed + self._state.current_epoch
        dataset = dataset.shuffle(buffer_size=self.buffer_size, seed=effective_seed)
        
        return dataset
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over the dataset, yielding (input_ids, target_ids) pairs.
        
        Uses a token buffer to create fixed-length sequences from
        variable-length documents.
        
        Yields:
            Tuple of (input_ids, target_ids), each of shape [max_length]
        """
        dataset = self._create_hf_dataset()
        
        # Token buffer for creating fixed-length sequences
        token_buffer = []
        samples_processed = 0
        sequences_yielded = 0
        
        for sample_idx, item in enumerate(dataset):
            # Check sample limit
            if self.num_samples is not None and samples_processed >= self.num_samples:
                break
            
            # Language filter
            if self.language_filter and item.get("language") != self.language_filter:
                continue
            
            # Train/val split filter
            if not self._should_include_sample(sample_idx):
                continue
            
            # Format text
            text = self._format_sample(item)
            if text is None:
                continue
            
            samples_processed += 1
            
            # Tokenize on-the-fly
            # Note: User must replace <|endoftext|> with actual token
            tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
            tokens.append(self.eos_token_id)  # Add document separator
            token_buffer.extend(tokens)
            
            # Yield fixed-length chunks when buffer is full
            while len(token_buffer) >= self.max_length + 1:
                chunk = token_buffer[:self.max_length + 1]
                token_buffer = token_buffer[self.max_length:]  # Non-overlapping stride
                
                sequences_yielded += 1
                
                # Skip sequences if resuming from checkpoint
                if sequences_yielded <= self._skip_sequences:
                    continue
                
                # Update state
                self._state.sequences_yielded = sequences_yielded
                self._state.samples_processed = samples_processed
                
                # Create tensors
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                target_ids = torch.tensor(chunk[1:], dtype=torch.long)
                
                yield input_ids, target_ids
        
        # Reset skip counter after full iteration
        self._skip_sequences = 0
    
    def set_epoch(self, epoch: int):
        """
        Set the current epoch for shuffle seed variation.
        
        Call this at the start of each epoch to get different shuffling.
        
        Args:
            epoch: Current epoch number (0-indexed)
        """
        self._state.current_epoch = epoch
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Return state dictionary for checkpointing.
        
        Used by StatefulDataLoader for mid-epoch checkpointing.
        
        Returns:
            Dictionary containing dataset state
        """
        return {
            "dataset_state": self._state.to_dict(),
            "skip_sequences": self._state.sequences_yielded,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load state from checkpoint.
        
        Used by StatefulDataLoader for resuming from mid-epoch checkpoint.
        
        Args:
            state_dict: State dictionary from previous state_dict() call
        """
        if "dataset_state" in state_dict:
            self._state = DatasetState.from_dict(state_dict["dataset_state"])
        
        # Set skip count to resume from correct position
        self._skip_sequences = state_dict.get("skip_sequences", 0)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current dataset statistics.
        
        Returns:
            Dictionary with statistics about processed data
        """
        return {
            "sequences_yielded": self._state.sequences_yielded,
            "samples_processed": self._state.samples_processed,
            "current_epoch": self._state.current_epoch,
            "split": self.split,
            "max_length": self.max_length,
            "buffer_size": self.buffer_size,
        }


class StreamingGPTDatasetWithSkip(StreamingGPTDataset):
    """
    Extended streaming dataset with explicit skip support for resuming.
    
    This version allows explicit setting of skip count at initialization,
    useful when not using StatefulDataLoader.
    
    Args:
        skip_sequences: Number of sequences to skip at start (for resuming)
        **kwargs: All arguments from StreamingGPTDataset
        
    Example:
        >>> # Resume from sequence 1000
        >>> dataset = StreamingGPTDatasetWithSkip(skip_sequences=1000, max_length=256)
    """
    
    def __init__(self, skip_sequences: int = 0, **kwargs):
        super().__init__(**kwargs)
        self._skip_sequences = skip_sequences
        self._initial_skip = skip_sequences
    
    def reset_skip(self):
        """Reset skip counter to initial value"""
        self._skip_sequences = self._initial_skip

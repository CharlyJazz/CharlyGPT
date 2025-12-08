"""
Checkpoint Manager for Streaming Datasets

Handles saving and loading dataset state for resuming training
from the exact position where it was interrupted.
"""

import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
from pathlib import Path

from .dataloader_factory import HAS_STATEFUL_DATALOADER

if HAS_STATEFUL_DATALOADER:
    from torchdata.stateful_dataloader import StatefulDataLoader


@dataclass
class CheckpointState:
    """
    Complete state for resuming training.
    
    Contains all information needed to resume training from
    the exact position where it was interrupted.
    """
    # Training progress
    global_step: int = 0
    epoch: int = 0
    batch_idx: int = 0
    
    # Loss tracking
    train_loss: float = 0.0
    val_loss: float = float("inf")
    best_val_loss: float = float("inf")
    
    # DataLoader state (for StatefulDataLoader)
    dataloader_state: Optional[Dict[str, Any]] = None
    
    # Dataset state (for manual skip)
    sequences_yielded: int = 0
    samples_processed: int = 0
    
    # Metadata
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for saving"""
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "batch_idx": self.batch_idx,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "best_val_loss": self.best_val_loss,
            "dataloader_state": self.dataloader_state,
            "sequences_yielded": self.sequences_yielded,
            "samples_processed": self.samples_processed,
            "config": self.config,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CheckpointState":
        """Create from dictionary"""
        return cls(
            global_step=d.get("global_step", 0),
            epoch=d.get("epoch", 0),
            batch_idx=d.get("batch_idx", 0),
            train_loss=d.get("train_loss", 0.0),
            val_loss=d.get("val_loss", float("inf")),
            best_val_loss=d.get("best_val_loss", float("inf")),
            dataloader_state=d.get("dataloader_state"),
            sequences_yielded=d.get("sequences_yielded", 0),
            samples_processed=d.get("samples_processed", 0),
            config=d.get("config", {}),
        )


class DatasetCheckpointManager:
    """
    Manager for saving and restoring dataset checkpoint state.
    
    Handles both StatefulDataLoader (automatic) and manual skip (fallback).
    
    Args:
        dataloader: The DataLoader to manage checkpoints for
        
    Example:
        >>> manager = DatasetCheckpointManager(train_loader)
        >>> 
        >>> # Save state mid-training
        >>> state = manager.get_state(global_step=1000, epoch=0)
        >>> 
        >>> # Later, restore state
        >>> manager.restore_state(state)
    """
    
    def __init__(
        self, 
        dataloader: Union[DataLoader, "StatefulDataLoader"],
    ):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        
        # Check if using StatefulDataLoader
        self._is_stateful = HAS_STATEFUL_DATALOADER and hasattr(dataloader, "state_dict")
        
        # Track iteration state manually as backup
        self._current_batch_idx = 0
    
    @property
    def is_stateful(self) -> bool:
        """Whether the dataloader supports native state management"""
        return self._is_stateful
    
    def get_state(
        self,
        global_step: int = 0,
        epoch: int = 0,
        batch_idx: int = 0,
        train_loss: float = 0.0,
        val_loss: float = float("inf"),
        best_val_loss: float = float("inf"),
        config: Optional[Dict[str, Any]] = None,
    ) -> CheckpointState:
        """
        Get current checkpoint state.
        
        Args:
            global_step: Current training step
            epoch: Current epoch
            batch_idx: Current batch index within epoch
            train_loss: Current training loss
            val_loss: Current validation loss
            best_val_loss: Best validation loss so far
            config: Optional config dict to save
        
        Returns:
            CheckpointState with all state information
        """
        state = CheckpointState(
            global_step=global_step,
            epoch=epoch,
            batch_idx=batch_idx,
            train_loss=train_loss,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            config=config or {},
        )
        
        # Get dataloader state if available
        if self._is_stateful:
            try:
                state.dataloader_state = self.dataloader.state_dict()
            except Exception as e:
                print(f"Warning: Could not get dataloader state: {e}")
        
        # Get dataset state for manual skip fallback
        if hasattr(self.dataset, "state_dict"):
            dataset_state = self.dataset.state_dict()
            state.sequences_yielded = dataset_state.get("sequences_yielded", 0)
            state.samples_processed = dataset_state.get("samples_processed", 0)
        elif hasattr(self.dataset, "_state"):
            state.sequences_yielded = self.dataset._state.sequences_yielded
            state.samples_processed = self.dataset._state.samples_processed
        
        return state
    
    def restore_state(self, state: CheckpointState) -> bool:
        """
        Restore checkpoint state.
        
        Args:
            state: CheckpointState to restore
        
        Returns:
            True if successfully restored with StatefulDataLoader,
            False if using manual skip fallback
        """
        # Try StatefulDataLoader first
        if self._is_stateful and state.dataloader_state is not None:
            try:
                self.dataloader.load_state_dict(state.dataloader_state)
                print(f"[OK] Restored dataloader state (StatefulDataLoader)")
                return True
            except Exception as e:
                print(f"Warning: Could not restore dataloader state: {e}")
        
        # Fallback to manual skip
        if hasattr(self.dataset, "load_state_dict"):
            self.dataset.load_state_dict({
                "sequences_yielded": state.sequences_yielded,
                "skip_sequences": state.sequences_yielded,
            })
            print(f"[OK] Restored dataset state (manual skip: {state.sequences_yielded} sequences)")
            return False
        elif hasattr(self.dataset, "_skip_sequences"):
            self.dataset._skip_sequences = state.sequences_yielded
            print(f"[OK] Set skip sequences: {state.sequences_yielded}")
            return False
        
        print("Warning: Could not restore any state - starting from beginning")
        return False
    
    def save_to_file(
        self,
        filepath: Union[str, Path],
        state: CheckpointState,
    ):
        """
        Save checkpoint state to file.
        
        Args:
            filepath: Path to save checkpoint
            state: CheckpointState to save
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as torch checkpoint
        torch.save(state.to_dict(), filepath)
        print(f"[OK] Saved checkpoint state to {filepath}")
    
    def load_from_file(
        self,
        filepath: Union[str, Path],
    ) -> CheckpointState:
        """
        Load checkpoint state from file.
        
        Args:
            filepath: Path to checkpoint file
        
        Returns:
            CheckpointState loaded from file
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        data = torch.load(filepath, weights_only=False)
        state = CheckpointState.from_dict(data)
        
        print(f"[OK] Loaded checkpoint state from {filepath}")
        print(f"  - Global step: {state.global_step}")
        print(f"  - Epoch: {state.epoch}")
        print(f"  - Sequences yielded: {state.sequences_yielded}")
        
        return state
    
    def get_resume_info(self, state: CheckpointState) -> Dict[str, Any]:
        """
        Get information about resuming from a checkpoint.
        
        Args:
            state: CheckpointState to analyze
        
        Returns:
            Dictionary with resume information
        """
        return {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "batch_idx": state.batch_idx,
            "sequences_to_skip": state.sequences_yielded,
            "has_dataloader_state": state.dataloader_state is not None,
            "is_stateful_loader": self._is_stateful,
            "resume_method": "stateful" if (self._is_stateful and state.dataloader_state) else "skip",
        }

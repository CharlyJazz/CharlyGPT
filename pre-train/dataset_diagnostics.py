"""
Dataset Diagnostics Module

Comprehensive logging to detect if dataset restarts on checkpoint resume.
Logs sample indices, content hashes, timestamps, and training state.

Usage:
    from dataset_diagnostics import DatasetDiagnostics
    
    diagnostics = DatasetDiagnostics(log_dir="path/to/experiment")
    
    # During training loop:
    diagnostics.log_batch(global_step, batch_idx, input_batch, sample_info)
    
    # On checkpoint save:
    diagnostics.log_checkpoint_save(global_step, sequences_yielded)
    
    # On checkpoint resume:
    diagnostics.log_checkpoint_resume(global_step, sequences_yielded, first_samples)
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import torch


class DatasetDiagnostics:
    """
    Diagnostic logger to track dataset position and detect restarts.
    
    Creates two log files:
    1. dataset_samples.jsonl - Detailed per-batch sample info
    2. dataset_checkpoints.jsonl - Checkpoint save/resume events
    """
    
    def __init__(self, log_dir: str, max_samples_per_log: int = 5):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.samples_log = self.log_dir / "dataset_samples.jsonl"
        self.checkpoints_log = self.log_dir / "dataset_checkpoints.jsonl"
        self.summary_log = self.log_dir / "dataset_summary.jsonl"
        
        self.max_samples_per_log = max_samples_per_log
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ring buffer to keep last N samples for comparison
        self.recent_samples: List[Dict[str, Any]] = []
        self.max_recent = 100
        
        # Log session start
        self._log_event(self.summary_log, {
            "event": "session_start",
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
        })
    
    def _compute_hash(self, tensor: torch.Tensor) -> str:
        """Compute a short hash of tensor content for comparison."""
        data = tensor.cpu().numpy().tobytes()
        return hashlib.md5(data).hexdigest()[:12]
    
    def _log_event(self, log_file: Path, event: Dict[str, Any]):
        """Append event to JSONL log file."""
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    
    def log_batch(
        self,
        global_step: int,
        batch_idx: int,
        input_batch: torch.Tensor,
        sequences_yielded: int = 0,
        samples_processed: int = 0,
        extra_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Log batch information during training.
        
        Args:
            global_step: Current training step
            batch_idx: Batch index within current epoch/run
            input_batch: Input tensor [batch_size, seq_len]
            sequences_yielded: Total sequences yielded by dataset
            samples_processed: Total raw samples processed
            extra_info: Additional info to log
        """
        batch_size = input_batch.shape[0]
        
        # Compute hashes for each sample in batch
        sample_hashes = []
        first_tokens = []
        for i in range(min(batch_size, self.max_samples_per_log)):
            sample = input_batch[i]
            sample_hashes.append(self._compute_hash(sample))
            first_tokens.append(sample[:10].tolist())
        
        event = {
            "event": "batch",
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "global_step": global_step,
            "batch_idx": batch_idx,
            "batch_size": batch_size,
            "sequences_yielded": sequences_yielded,
            "samples_processed": samples_processed,
            "sample_hashes": sample_hashes,
            "first_tokens": first_tokens,
        }
        
        if extra_info:
            event["extra"] = extra_info
        
        # Log every 100 steps to avoid huge files
        if global_step % 100 == 0:
            self._log_event(self.samples_log, event)
        
        # Always keep in memory for comparison
        self.recent_samples.append({
            "global_step": global_step,
            "hashes": sample_hashes,
            "first_tokens": first_tokens,
        })
        if len(self.recent_samples) > self.max_recent:
            self.recent_samples.pop(0)
    
    def log_checkpoint_save(
        self,
        global_step: int,
        sequences_yielded: int,
        samples_processed: int,
        dataloader_state_saved: bool,
    ):
        """Log checkpoint save event with dataset state."""
        # Get last N sample hashes for comparison on resume
        last_samples = self.recent_samples[-10:] if self.recent_samples else []
        
        event = {
            "event": "checkpoint_save",
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "global_step": global_step,
            "sequences_yielded": sequences_yielded,
            "samples_processed": samples_processed,
            "dataloader_state_saved": dataloader_state_saved,
            "last_sample_hashes": [s["hashes"] for s in last_samples],
            "last_sample_steps": [s["global_step"] for s in last_samples],
        }
        
        self._log_event(self.checkpoints_log, event)
        self._log_event(self.summary_log, event)
        
        print(f"[DIAG] Checkpoint saved at step {global_step}")
        print(f"       sequences_yielded: {sequences_yielded}")
        print(f"       samples_processed: {samples_processed}")
        print(f"       dataloader_state_saved: {dataloader_state_saved}")
    
    def log_checkpoint_resume(
        self,
        global_step: int,
        sequences_yielded: int,
        samples_processed: int,
        skip_sequences: int,
        dataloader_state_restored: bool,
        first_batch_after_resume: Optional[torch.Tensor] = None,
    ):
        """
        Log checkpoint resume event and compare with saved state.
        
        This is the KEY diagnostic - if first_batch_after_resume hashes
        match samples from early in training (not from checkpoint save),
        the dataset is restarting!
        """
        first_hashes = []
        first_tokens = []
        if first_batch_after_resume is not None:
            for i in range(min(first_batch_after_resume.shape[0], 5)):
                sample = first_batch_after_resume[i]
                first_hashes.append(self._compute_hash(sample))
                first_tokens.append(sample[:10].tolist())
        
        event = {
            "event": "checkpoint_resume",
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "global_step": global_step,
            "sequences_yielded": sequences_yielded,
            "samples_processed": samples_processed,
            "skip_sequences": skip_sequences,
            "dataloader_state_restored": dataloader_state_restored,
            "first_batch_hashes": first_hashes,
            "first_batch_tokens": first_tokens,
        }
        
        self._log_event(self.checkpoints_log, event)
        self._log_event(self.summary_log, event)
        
        print(f"\n{'='*60}")
        print(f"[DIAG] CHECKPOINT RESUME DIAGNOSTICS")
        print(f"{'='*60}")
        print(f"  Resuming from step: {global_step}")
        print(f"  sequences_yielded in checkpoint: {sequences_yielded}")
        print(f"  skip_sequences set to: {skip_sequences}")
        print(f"  dataloader_state_restored: {dataloader_state_restored}")
        print(f"  First batch hashes after resume: {first_hashes}")
        print(f"  First tokens: {first_tokens}")
        print(f"{'='*60}\n")
    
    def compare_with_previous_session(self) -> Dict[str, Any]:
        """
        Compare current session's first samples with previous sessions.
        
        Returns analysis of whether dataset appears to be restarting.
        """
        if not self.samples_log.exists():
            return {"status": "no_previous_data"}
        
        # Load all previous batch logs
        previous_hashes = {}
        with open(self.samples_log, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                    if event.get("event") == "batch":
                        step = event["global_step"]
                        hashes = event.get("sample_hashes", [])
                        session = event.get("session_id", "unknown")
                        for h in hashes:
                            if h not in previous_hashes:
                                previous_hashes[h] = []
                            previous_hashes[h].append({
                                "step": step,
                                "session": session,
                            })
                except:
                    continue
        
        # Check current session's samples against previous
        duplicates = []
        for sample in self.recent_samples:
            for h in sample["hashes"]:
                if h in previous_hashes:
                    for prev in previous_hashes[h]:
                        if prev["session"] != self.session_id:
                            duplicates.append({
                                "hash": h,
                                "current_step": sample["global_step"],
                                "previous_step": prev["step"],
                                "previous_session": prev["session"],
                            })
        
        return {
            "status": "analyzed",
            "total_unique_hashes": len(previous_hashes),
            "duplicates_found": len(duplicates),
            "duplicate_details": duplicates[:20],  # First 20
            "likely_restart": len(duplicates) > 10,
        }
    
    def get_diagnostic_summary(self) -> str:
        """Generate human-readable diagnostic summary."""
        analysis = self.compare_with_previous_session()
        
        lines = [
            "\n" + "="*60,
            "DATASET DIAGNOSTICS SUMMARY",
            "="*60,
            f"Session ID: {self.session_id}",
            f"Samples in memory: {len(self.recent_samples)}",
            f"Log directory: {self.log_dir}",
            "",
        ]
        
        if analysis["status"] == "analyzed":
            lines.extend([
                f"Unique sample hashes in logs: {analysis['total_unique_hashes']}",
                f"Duplicate samples found: {analysis['duplicates_found']}",
                "",
            ])
            
            if analysis["likely_restart"]:
                lines.extend([
                    "⚠️  WARNING: DATASET LIKELY RESTARTING ON RESUME!",
                    "   Many samples match previous training sessions.",
                    "   This means you're re-training on the same data.",
                    "",
                ])
            else:
                lines.append("✓ No significant duplicates detected.")
        
        lines.append("="*60 + "\n")
        return "\n".join(lines)


# Singleton instance for easy access
_diagnostics_instance: Optional[DatasetDiagnostics] = None


def get_diagnostics(log_dir: Optional[str] = None) -> Optional[DatasetDiagnostics]:
    """Get or create the global diagnostics instance."""
    global _diagnostics_instance
    if _diagnostics_instance is None and log_dir:
        _diagnostics_instance = DatasetDiagnostics(log_dir)
    return _diagnostics_instance


def init_diagnostics(log_dir: str) -> DatasetDiagnostics:
    """Initialize the global diagnostics instance."""
    global _diagnostics_instance
    _diagnostics_instance = DatasetDiagnostics(log_dir)
    return _diagnostics_instance

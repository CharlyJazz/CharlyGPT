"""
Pre-training script for GPT-2 from scratch
Using HuggingFace SYNTH dataset
"""

import torch
import torch.nn as nn
import sys
import os
import signal
import yaml
from pathlib import Path
from typing import Optional

# Global flag for graceful shutdown
_shutdown_requested = False

def _signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    global _shutdown_requested
    print("\n\n⚠️  Shutdown signal received. Finishing current operation...")
    _shutdown_requested = True

# Register signal handlers
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arch.gpt_model import GPTModel
from arch.config import GPT_CONFIG_124M
from utils import tokenizer, text_to_token_ids, token_ids_to_text, generate_text_simple

# Streaming dataset imports
from dataset import (
    create_streaming_dataloaders,
    DatasetCheckpointManager,
    HAS_STATEFUL_DATALOADER,
    print_dataloader_status,
)

# MLflow for experiment tracking
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    mlflow = None

# ============================================================================
# MLFLOW TRACKING
# ============================================================================

MLFLOW_TRACKING_URI = "file:./mlruns"

# Global to store current run_id for checkpointing
_current_mlflow_run_id: Optional[str] = None


def setup_mlflow_tracking(experiment_name: str, config: dict, resume_run_id: Optional[str] = None) -> bool:
    """
    Initialize MLflow tracking for the training run.
    
    Args:
        experiment_name: Name of the MLflow experiment
        config: Training configuration dict
        resume_run_id: If provided, resume this existing run instead of creating new one
    
    Returns:
        True if MLflow is available and initialized, False otherwise
    """
    global _current_mlflow_run_id
    
    if not HAS_MLFLOW:
        print("[WARN] MLflow not installed. Metrics will not be logged.")
        print("       Install with: pip install mlflow")
        return False
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)  # Same experiment as mlflow_viewer.py
    
    # Resume existing run or start new one
    run_description = 'Main train run to log useful information'
    if resume_run_id:
        try:
            mlflow.start_run(run_id=resume_run_id, description=run_description)
            _current_mlflow_run_id = resume_run_id
            print(f"[OK] MLflow: Resumed existing run {resume_run_id[:8]}...")
        except Exception as e:
            print(f"[WARN] Could not resume MLflow run {resume_run_id}: {e}")
            print("       Starting new run instead.")
            mlflow.start_run(run_name=f"train-{experiment_name}", description=run_description)
            _current_mlflow_run_id = mlflow.active_run().info.run_id
    else:
        mlflow.start_run(run_name=f"train-{experiment_name}", description=run_description)
        _current_mlflow_run_id = mlflow.active_run().info.run_id
        
        # Log hyperparameters only for new runs
        mlflow.log_params({
            "learning_rate": config["learning_rate"],
            "min_learning_rate": config.get("min_learning_rate", config["learning_rate"] * 0.1),
            "batch_size": config["batch_size"],
            "max_length": config["max_length"],
            "num_epochs": config["num_epochs"],
            "warmup_steps": config["warmup_steps"],
            "total_steps": config.get("total_steps", "unlimited"),
            "weight_decay": config["weight_decay"],
            "gradient_clip": config["gradient_clip"],
            "num_samples": config["num_samples"],
        })
        print(f"[OK] MLflow: New run {_current_mlflow_run_id[:8]}...")
    
    print(f"     Tracking URI: {MLFLOW_TRACKING_URI}")
    return True


def get_mlflow_run_id() -> Optional[str]:
    """Get the current MLflow run ID for saving in checkpoints"""
    return _current_mlflow_run_id


def log_metrics_to_mlflow(step: int, metrics: dict):
    """Log metrics to MLflow if available"""
    if HAS_MLFLOW and mlflow.active_run():
        mlflow.log_metrics(metrics, step=step)


def end_mlflow_tracking():
    """End the MLflow run"""
    if HAS_MLFLOW and mlflow.active_run():
        mlflow.end_run()


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

EXPERIMENT_FILE = r"C:\Users\Usuario\CascadeProjects\windsurf-project\pre-train\experiments\SmallGPT2-Samples2M.yaml"


def load_training_config(config_path: str) -> dict:
    """Load training configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Flatten nested config into single dict for compatibility
    training_config = {
        "experiment_name": config.get("experiment_name", "default_experiment"),
        # Data
        "num_samples": config.get("data", {}).get("num_samples", 100000),
        "max_length": config.get("data", {}).get("max_length", 256),
        # Training
        "batch_size": config.get("training", {}).get("batch_size", 8),
        "num_epochs": config.get("training", {}).get("num_epochs", 2),
        "learning_rate": config.get("training", {}).get("learning_rate", 3e-4),
        "weight_decay": config.get("training", {}).get("weight_decay", 0.1),
        "gradient_clip": config.get("training", {}).get("gradient_clip", 1.0),
        "warmup_steps": config.get("training", {}).get("warmup_steps", 2000),
        "total_steps": config.get("training", {}).get("total_steps", None),
        "min_learning_rate": config.get("training", {}).get("min_learning_rate", None),
        # Evaluation
        "eval_freq": config.get("evaluation", {}).get("eval_freq", 500),
        "eval_iters": config.get("evaluation", {}).get("eval_iters", 20),
        "save_every_n_iterations": config.get("evaluation", {}).get("save_every_n_iterations", 1000),
        # Storage
        "base_folder": config.get("storage", {}).get("base_folder", "."),
        "checkpoint_to_resume": config.get("storage", {}).get("checkpoint_to_resume"),
        # Hardware
        "device": config.get("hardware", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu",
        # Streaming dataset
        "buffer_size": config.get("data", {}).get("buffer_size", 10000),
        "seed": config.get("data", {}).get("seed", 42),
        "train_ratio": config.get("data", {}).get("train_ratio", 0.9),
    }
    
    return training_config


# Load config from YAML
TRAINING_CONFIG = load_training_config(EXPERIMENT_FILE)


# ============================================================================
# RUNTIME INFORMATION
# ============================================================================

def generate_runtime_information(model: GPTModel, config: dict, config_path: str):
    """
    Genera y escribe información de runtime en el archivo YAML del experimento.
    
    Solo escribe valores EXACTOS (no estimaciones):
        - model_parameters: Total de parámetros del modelo
        - trainable_parameters: Parámetros entrenables
        - tokens_per_batch: batch_size × max_length
    
    Args:
        model: El modelo GPT inicializado
        config: Training configuration dict
        config_path: Ruta al archivo YAML del experimento
    """
    # Calcular parámetros del modelo (exactos)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokens_per_batch = config["batch_size"] * config["max_length"]
    
    # Leer YAML existente
    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_content = yaml.safe_load(f)
    
    # Agregar/actualizar sección runtime (solo valores exactos)
    yaml_content['runtime'] = {
        'model_parameters': total_params,
        'trainable_parameters': trainable_params,
        'tokens_per_batch': tokens_per_batch,
        # Nota: iterations_expected y batches_per_epoch no se pueden calcular
        # con streaming porque dependen de la tokenización on-the-fly
    }
    
    # Escribir YAML actualizado
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    # Imprimir resumen
    print("\n" + "="*60)
    print("RUNTIME INFORMATION (guardado en YAML)")
    print("="*60)
    print(f"  Model Parameters:      {total_params:,}")
    print(f"  Trainable Parameters:  {trainable_params:,}")
    print(f"  Tokens per Batch:      {tokens_per_batch:,}")
    print(f"  (iterations unknown with streaming - counted at runtime)")
    print("="*60 + "\n")


# ============================================================================
# LEARNING RATE SCHEDULING
# ============================================================================

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float,
    max_lr: float,
    last_step: int = -1
):
    """
    Create a learning rate scheduler with linear warmup + cosine decay.
    
    Schedule:
        Step 0 → warmup_steps:     Linear warmup (1% → 100% of max_lr)
        Step warmup_steps → total: Cosine decay (max_lr → min_lr)
    
    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr: Minimum LR at end of training (typically 10% of max_lr)
        max_lr: Maximum LR (after warmup)
        last_step: Last completed step (for resuming). -1 = start fresh.
    
    Returns:
        LR scheduler (SequentialLR combining warmup + cosine)
    """
    # Warmup: 1% → 100% of base LR over warmup_steps
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # Cosine decay: max_lr → min_lr over remaining steps
    # Note: CosineAnnealingLR uses eta_min as the minimum LR
    decay_steps = max(1, total_steps - warmup_steps)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=decay_steps,
        eta_min=min_lr
    )
    
    # Combine: warmup first, then cosine
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    # Fast-forward scheduler if resuming from checkpoint
    if last_step > 0:
        for _ in range(last_step):
            scheduler.step()
    
    return scheduler


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer"""
    return optimizer.param_groups[0]['lr']


# ============================================================================
# LOSS CALCULATION
# ============================================================================

def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor, model: GPTModel, device: torch.device) -> torch.Tensor:
    """Calculate loss for a single batch"""
    input_batch: torch.Tensor = input_batch.to(device)
    target_batch: torch.Tensor = target_batch.to(device)
    logits: torch.Tensor = model(input_batch)
    loss: torch.Tensor = nn.functional.cross_entropy(
        logits.flatten(0, 1), 
        target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Calculate average loss over multiple batches (streaming-compatible)"""
    total_loss = 0.0
    batches_counted = 0
    
    # For streaming datasets, we can't use len() - just iterate up to num_batches
    if num_batches is None:
        num_batches = 100  # Default limit for streaming
    
    model.eval()
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            batches_counted += 1
    
    model.train()
    return total_loss / max(batches_counted, 1)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, train_loader, val_loader, device, eval_iters):
    """Evaluate model on train and validation sets"""
    train_loss = calc_loss_loader(train_loader, model, device, eval_iters)
    val_loss = calc_loss_loader(val_loader, model, device, eval_iters)
    return train_loss, val_loss


def generate_sample(model, tokenizer, device, prompt="Once upon a time"):
    """Generate text sample to monitor progress"""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(prompt, tokenizer).to(device)
    
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=30,
            context_size=context_size,
            temperature=0.7,
            top_k=50
        )
    
    text = token_ids_to_text(token_ids, tokenizer)
    model.train()
    return text


# ============================================================================
# TRAINING LOOP
# ============================================================================

def validate_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    """
    Validate checkpoint file exists and can be loaded BEFORE dataset preparation.
    Returns the checkpoint dict if valid, raises exception otherwise.
    """
    print("\n" + "="*60)
    print("VALIDATING CHECKPOINT")
    print("="*60)
    print(f"Checking checkpoint: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    try:
        checkpoint: dict = torch.load(checkpoint_path, map_location=device)
        
        # Validate required keys
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'global_step', 'epoch']
        missing_keys = [k for k in required_keys if k not in checkpoint]
        if missing_keys:
            raise KeyError(f"Checkpoint missing required keys: {missing_keys}")
        
        print(f"\n  ✓ Checkpoint is valid!")
        print(f"  - Global step: {checkpoint['global_step']:,}")
        print(f"  - Epoch: {checkpoint['epoch'] + 1}")
        print(f"  - Val loss: {checkpoint.get('val_loss', 'N/A')}")
        print("="*60 + "\n")
        
        return checkpoint
        
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")


def load_checkpoint(checkpoint: dict, model: GPTModel, optimizer: torch.optim.Optimizer, device: torch.device):
    """
    Load a pre-validated checkpoint and restore training state.
    Returns: (global_step, start_epoch, best_val_loss, batches_per_epoch)
    """
    print("\n" + "="*60)
    print("RESUMING FROM CHECKPOINT")
    print("="*60)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    global_step = checkpoint['global_step']
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    train_loss = checkpoint.get('train_loss', 'N/A')
    val_loss = checkpoint.get('val_loss', 'N/A')
    batches_per_epoch = checkpoint.get('batches_per_epoch', 0)
    
    print(f"\n  Checkpoint info:")
    print(f"  - Global step: {global_step:,}")
    print(f"  - Epoch: {epoch + 1}")
    print(f"  - Batches per epoch: {batches_per_epoch:,}")
    print(f"  - Train loss at save: {train_loss}")
    print(f"  - Val loss at save: {val_loss}")
    print(f"  - Best val loss so far: {best_val_loss:.4f}")
    print(f"\n  Resuming training from step {global_step + 1}...")
    print("="*60 + "\n")
    
    return global_step, epoch, best_val_loss, batches_per_epoch


def save_checkpoint(model: GPTModel, optimizer: torch.optim.Optimizer, epoch: int, global_step: int, 
                    train_loss: float, val_loss: float, best_val_loss: float, checkpoint_dir: Path,
                    dataloader_state: Optional[dict] = None, sequences_yielded: int = 0):
    """Save a training checkpoint with error handling (streaming-compatible)"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{global_step}.pt"
    temp_path = checkpoint_dir / f"checkpoint_step_{global_step}.pt.tmp"
    
    checkpoint_data = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        # Streaming dataset state
        'dataloader_state': dataloader_state,
        'sequences_yielded': sequences_yielded,
        # MLflow run ID for resuming the same run
        'mlflow_run_id': get_mlflow_run_id(),
    }
    
    try:
        # Save to temp file first, then rename (atomic operation)
        torch.save(checkpoint_data, temp_path)
        
        # Remove old checkpoint if exists, then rename temp to final
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        temp_path.rename(checkpoint_path)
        
        print(f"\n  ✓ Checkpoint saved: {checkpoint_path}")
        
    except RuntimeError as e:
        print(f"\n  ⚠️  WARNING: Failed to save checkpoint at step {global_step}")
        print(f"     Error: {e}")
        print(f"     Possible causes: disk full, permissions, or antivirus interference")
        print(f"     Training will continue. Check disk space!")
        
        # Clean up temp file if it exists
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass
    
    # Keep only last 3 checkpoints to save disk space
    # Sort numerically by step number, not alphabetically
    checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
    checkpoints.sort(key=lambda p: int(p.stem.split("_")[-1]))  # Extract step number for sorting
    if len(checkpoints) > 3:
        for old_ckpt in checkpoints[:-3]:
            old_ckpt.unlink()
            print(f"  Removed old checkpoint: {old_ckpt.name}")


def train_model(model: GPTModel, train_loader, val_loader, optimizer: torch.optim.Optimizer, config: dict):
    """Main training loop (streaming-compatible)"""
    device: torch.device = config["device"]
    model.to(device)
    model.train()
    
    # Setup experiment directory: base_folder / experiment_name / checkpoints
    base_folder = Path(config.get("base_folder", "."))
    experiment_name = config.get("experiment_name", "default_experiment")
    experiment_dir = base_folder / experiment_name
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize training state
    global_step = 0
    start_epoch = 0
    best_val_loss = float('inf')
    sequences_yielded = 0
    
    # Checkpoint manager for streaming dataloader state
    checkpoint_manager = DatasetCheckpointManager(train_loader)
    
    # Load checkpoint if provided (already validated in main())
    validated_checkpoint = config.get("_validated_checkpoint")
    if validated_checkpoint is not None:
        global_step, start_epoch, best_val_loss, _ = load_checkpoint(
            validated_checkpoint, model, optimizer, device
        )
        # Restore dataloader state if available
        if HAS_STATEFUL_DATALOADER and validated_checkpoint.get('dataloader_state'):
            try:
                train_loader.load_state_dict(validated_checkpoint['dataloader_state'])
                print(f"  [OK] Restored dataloader state (StatefulDataLoader)")
            except Exception as e:
                print(f"  [WARN] Could not restore dataloader state: {e}")
        elif validated_checkpoint.get('sequences_yielded', 0) > 0:
            # Fallback: set skip on dataset
            sequences_yielded = validated_checkpoint['sequences_yielded']
            if hasattr(train_loader.dataset, '_skip_sequences'):
                train_loader.dataset._skip_sequences = sequences_yielded
                print(f"  [OK] Will skip {sequences_yielded} sequences to resume")
    
    # Create LR scheduler with warmup + cosine decay
    total_steps = config.get("total_steps")
    min_lr = config.get("min_learning_rate") or config["learning_rate"] * 0.1
    max_lr = config["learning_rate"]
    warmup_steps = config["warmup_steps"]
    
    if total_steps:
        scheduler = create_lr_scheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr,
            max_lr=max_lr,
            last_step=global_step  # Resume from checkpoint
        )
        lr_schedule_info = f"warmup {warmup_steps} → cosine decay to {min_lr:.2e} over {total_steps} steps"
    else:
        scheduler = None
        lr_schedule_info = f"warmup {warmup_steps} steps → {max_lr:.2e} (constant, no decay)"
    
    print(f"\nTraining on device: {device}")
    print(f"Streaming mode: ON (memory-efficient)")
    print(f"StatefulDataLoader: {'YES' if HAS_STATEFUL_DATALOADER else 'NO (install torchdata>=0.8.0)'}")
    print(f"LR Schedule: {lr_schedule_info}")
    print("-" * 60)
    
    for epoch in range(start_epoch, config["num_epochs"]):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{config['num_epochs']}")
        print(f"{'='*60}")
        
        # Set epoch for shuffle seed variation
        if hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)
        
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            # Check for shutdown signal
            global _shutdown_requested
            if _shutdown_requested:
                print(f"\n[WARN] Graceful shutdown at step {global_step}")
                print("  Saving emergency checkpoint...")
                try:
                    last_loss = loss.item()
                except:
                    last_loss = 0.0
                
                # Get dataloader state for resume
                dl_state = None
                if HAS_STATEFUL_DATALOADER:
                    try:
                        dl_state = train_loader.state_dict()
                    except:
                        pass
                
                save_checkpoint(
                    model, optimizer, epoch, global_step,
                    last_loss, best_val_loss, best_val_loss, checkpoint_dir,
                    dataloader_state=dl_state,
                    sequences_yielded=train_loader.dataset._state.sequences_yielded if hasattr(train_loader.dataset, '_state') else 0
                )
                print("  [OK] Emergency checkpoint saved. Exiting cleanly.")
                return
            
            # Forward pass
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config["gradient_clip"]
            )
            
            optimizer.step()
            
            # Update learning rate (after optimizer.step, as per PyTorch convention)
            if scheduler is not None:
                scheduler.step()
            
            global_step += 1
            current_lr = get_current_lr(optimizer)
            
            # Print training progress + log to MLflow every 100 steps
            if global_step % 10 == 0:
                print(f"Step {global_step:05d} | Batch {batch_idx:04d} | Loss: {loss.item():.4f} | LR: {current_lr:.2e}")
            
            if global_step % 100 == 0:
                log_metrics_to_mlflow(global_step, {
                    "train_loss_batch": loss.item(),
                    "learning_rate": current_lr,
                })
            
            # Evaluation
            if global_step % config["eval_freq"] == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, config["eval_iters"]
                )
                
                print(f"\n{'─'*60}")
                print(f"[EVAL] Step {global_step}")
                print(f"   Train Loss: {train_loss:.4f}")
                print(f"   Val Loss:   {val_loss:.4f}")
                print(f"   LR:         {current_lr:.2e}")
                
                # Log evaluation metrics to MLflow
                log_metrics_to_mlflow(global_step, {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                })
                
                # Generate sample
                sample_text = generate_sample(model, tokenizer, device)
                print(f"\n   Sample generation:")
                print(f"   {sample_text[:100]}...")
                print(f"{'─'*60}\n")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = checkpoint_dir / "best_model.pt"
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, best_model_path)
                    print(f"  [OK] Best model saved: {best_model_path} (val_loss: {val_loss:.4f})")
            
            # Save periodic checkpoint
            if global_step % config["save_every_n_iterations"] == 0:
                dl_state = None
                if HAS_STATEFUL_DATALOADER:
                    try:
                        dl_state = train_loader.state_dict()
                    except:
                        pass
                
                save_checkpoint(
                    model, optimizer, epoch, global_step,
                    loss.item(), best_val_loss, best_val_loss, checkpoint_dir,
                    dataloader_state=dl_state,
                    sequences_yielded=train_loader.dataset._state.sequences_yielded if hasattr(train_loader.dataset, '_state') else 0
                )
    
    # Save final checkpoint
    dl_state = None
    if HAS_STATEFUL_DATALOADER:
        try:
            dl_state = train_loader.state_dict()
        except:
            pass
    
    save_checkpoint(
        model, optimizer, epoch, global_step,
        loss.item(), best_val_loss, best_val_loss, checkpoint_dir,
        dataloader_state=dl_state,
        sequences_yielded=train_loader.dataset._state.sequences_yielded if hasattr(train_loader.dataset, '_state') else 0
    )
    print("[OK] Final checkpoint saved")
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("GPT-2 PRE-TRAINING FROM SCRATCH")
    print("Streaming Mode: Memory-efficient data loading")
    print("="*60 + "\n")
    
    # Show dataloader status
    print_dataloader_status()
    
    device = TRAINING_CONFIG["device"]
    
    # ========================================
    # STEP 1: Setup paths and validate checkpoint FIRST
    # ========================================
    base_folder = Path(TRAINING_CONFIG.get("base_folder", "."))
    experiment_name = TRAINING_CONFIG.get("experiment_name", "default_experiment")
    experiment_dir = base_folder / experiment_name
    checkpoint_dir = experiment_dir / "checkpoints"
    
    print(f"Experiment: {experiment_name}")
    print(f"Storage: {experiment_dir}\n")
    
    validated_checkpoint = None
    checkpoint_to_resume = TRAINING_CONFIG.get("checkpoint_to_resume")
    
    if checkpoint_to_resume is not None:
        checkpoint_path = checkpoint_dir / checkpoint_to_resume
        try:
            validated_checkpoint = validate_checkpoint(checkpoint_path, device)
            print("[OK] Checkpoint validated successfully.\n")
        except Exception as e:
            print(f"\n[ERROR] CHECKPOINT ERROR: {e}")
            print("\nAborting. Please fix checkpoint_to_resume or set it to None.\n")
            sys.exit(1)
    
    # Store validated checkpoint in config for later use
    TRAINING_CONFIG["_validated_checkpoint"] = validated_checkpoint
    
    # ========================================
    # STEP 1.5: Setup MLflow tracking
    # ========================================
    # Resume MLflow run if checkpoint has run_id, otherwise start new run
    resume_run_id = None
    if validated_checkpoint is not None:
        resume_run_id = validated_checkpoint.get('mlflow_run_id')
    
    setup_mlflow_tracking(experiment_name, TRAINING_CONFIG, resume_run_id=resume_run_id)
    
    # ========================================
    # STEP 2: Initialize model
    # ========================================
    print("Initializing model...")
    model = GPTModel(GPT_CONFIG_124M)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Model config: {GPT_CONFIG_124M}\n")
    
    # ========================================
    # STEP 2.5: Generate runtime information and save to YAML
    # ========================================
    generate_runtime_information(model, TRAINING_CONFIG, EXPERIMENT_FILE)
    
    # ========================================
    # STEP 3: Create streaming dataloaders (instant, no pre-loading)
    # ========================================
    print("\n" + "="*60)
    print("CREATING STREAMING DATALOADERS")
    print("="*60)
    print(f"  Dataset: PleIAs/SYNTH")
    print(f"  Max samples: {TRAINING_CONFIG['num_samples']:,}")
    print(f"  Max length: {TRAINING_CONFIG['max_length']}")
    print(f"  Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"  Buffer size: {TRAINING_CONFIG['buffer_size']}")
    print(f"  Train ratio: {TRAINING_CONFIG['train_ratio']}")
    
    train_loader, val_loader = create_streaming_dataloaders(
        batch_size=TRAINING_CONFIG["batch_size"],
        max_length=TRAINING_CONFIG["max_length"],
        num_samples=TRAINING_CONFIG["num_samples"],
        buffer_size=TRAINING_CONFIG["buffer_size"],
        seed=TRAINING_CONFIG["seed"],
        train_ratio=TRAINING_CONFIG["train_ratio"],
        num_workers=0,  # Streaming doesn't benefit from workers
        pin_memory=True,
        drop_last=True,
    )
    
    print("[OK] Streaming dataloaders created (no data loaded yet)")
    print("="*60 + "\n")
    
    # ========================================
    # STEP 4: Initialize optimizer
    # ========================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"]
    )
    
    # Train
    try:
        train_model(model, train_loader, val_loader, optimizer, TRAINING_CONFIG)
    finally:
        # Always close MLflow run, even on error/interrupt
        end_mlflow_tracking()


if __name__ == "__main__":
    main()

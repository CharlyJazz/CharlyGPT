"""
Pre-training script for GPT-2 from scratch
Using HuggingFace SYNTH dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, IterableDataset
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
from utils import tokenizer, text_to_token_ids, token_ids_to_text, generate_text_simple, create_dataloader_v1 

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
        "warmup_steps": config.get("training", {}).get("warmup_steps", 500),
        # Evaluation
        "eval_freq": config.get("evaluation", {}).get("eval_freq", 500),
        "eval_iters": config.get("evaluation", {}).get("eval_iters", 20),
        "save_every_n_iterations": config.get("evaluation", {}).get("save_every_n_iterations", 1000),
        # Storage
        "base_folder": config.get("storage", {}).get("base_folder", "."),
        "checkpoint_to_resume": config.get("storage", {}).get("checkpoint_to_resume"),
        # Hardware
        "device": config.get("hardware", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu",
    }
    
    return training_config


# Load config from YAML
TRAINING_CONFIG = load_training_config(EXPERIMENT_FILE)


# ============================================================================
# RUNTIME INFORMATION
# ============================================================================

def calculate_iterations_expected(config: dict, avg_tokens_per_sample: int = 750) -> dict:
    """
    Calcula el número esperado de iteraciones del training loop.
    
    Args:
        config: Training configuration dict
        avg_tokens_per_sample: Promedio de tokens por sample (Q + Reasoning + A)
    
    Returns:
        Dict con métricas calculadas
    """
    num_samples = config["num_samples"]
    batch_size  = config["batch_size"]
    max_length  = config["max_length"]
    num_epochs  = config["num_epochs"]
    
    # Estimación de tokens totales
    total_tokens_estimated = num_samples * avg_tokens_per_sample
    
    # Batches por epoch = total_tokens / (batch_size * max_length)
    # Cada batch tiene batch_size secuencias de max_length tokens
    tokens_per_batch = batch_size * max_length
    batches_per_epoch = total_tokens_estimated // tokens_per_batch
    
    # Total de iteraciones (steps)
    total_iterations = batches_per_epoch * num_epochs
    
    return {
        "total_tokens_estimated": total_tokens_estimated,
        "tokens_per_batch": tokens_per_batch,
        "batches_per_epoch": batches_per_epoch,
        "total_iterations": total_iterations,
        "avg_tokens_per_sample": avg_tokens_per_sample,
    }


def generate_runtime_information(model: GPTModel, config: dict, config_path: str):
    """
    Genera y escribe información de runtime en el archivo YAML del experimento.
    
    Escribe:
        - model_parameters: Total de parámetros del modelo
        - iterations_expected: Total de iteraciones esperadas del training loop
        - Otras métricas calculadas
    
    Args:
        model: El modelo GPT inicializado
        config: Training configuration dict
        config_path: Ruta al archivo YAML del experimento
    """
    # Calcular parámetros del modelo
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calcular iteraciones esperadas
    iterations_info = calculate_iterations_expected(config)
    
    # Leer YAML existente
    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_content = yaml.safe_load(f)
    
    # Agregar/actualizar sección runtime
    yaml_content['runtime'] = {
        'model_parameters': total_params,
        'trainable_parameters': trainable_params,
        'iterations_expected': iterations_info['total_iterations'],
        'batches_per_epoch': iterations_info['batches_per_epoch'],
        'total_tokens_estimated': iterations_info['total_tokens_estimated'],
        'tokens_per_batch': iterations_info['tokens_per_batch'],
        'avg_tokens_per_sample': iterations_info['avg_tokens_per_sample'],
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
    print(f"  Iterations Expected:   {iterations_info['total_iterations']:,}")
    print(f"  Batches per Epoch:     {iterations_info['batches_per_epoch']:,}")
    print(f"  Total Tokens (est.):   {iterations_info['total_tokens_estimated']:,}")
    print(f"  Tokens per Batch:      {iterations_info['tokens_per_batch']:,}")
    print("="*60 + "\n")


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
    """Calculate average loss over multiple batches"""
    total_loss = 0.0
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    model.eval()
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
    
    model.train()
    return total_loss / num_batches


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
                    batches_per_epoch: int = 0):
    """Save a training checkpoint with error handling"""
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
        'batches_per_epoch': batches_per_epoch,
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


def train_model(model: GPTModel, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer, config: dict):
    """Main training loop"""
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
    skip_batches = 0  # Number of batches to skip when resuming
    batches_per_epoch = len(train_loader)
    
    # Load checkpoint if provided (already validated in main())
    validated_checkpoint = config.get("_validated_checkpoint")
    if validated_checkpoint is not None:
        global_step, start_epoch, best_val_loss, saved_batches_per_epoch = load_checkpoint(
            validated_checkpoint, model, optimizer, device
        )
        # Calculate how many batches to skip in the current epoch
        if saved_batches_per_epoch > 0:
            batches_done_in_epoch = global_step % saved_batches_per_epoch
            skip_batches = batches_done_in_epoch
            print(f"  Will skip {skip_batches} batches in epoch {start_epoch + 1}")
    
    print(f"\nTraining on device: {device}")
    print(f"Total training batches: {len(train_loader)}")
    print(f"Total validation batches: {len(val_loader)}")
    print("-" * 60)
    
    # DEBUG: Inspect first batch
    print("\n" + "="*60)
    print("DEBUG: First batch inspection")
    print("="*60)
    first_input, first_target = next(iter(train_loader))
    print(f"Input shape: {first_input.shape}")
    print(f"Target shape: {first_target.shape}")
    print(f"\nFirst sequence input tokens: {first_input[0][:20].tolist()}...")
    print(f"First sequence target tokens: {first_target[0][:20].tolist()}...")
    print(f"\nDecoded input (first 200 chars):")
    print(tokenizer.decode(first_input[0].tolist())[:200])
    print(f"\nDecoded target (first 200 chars):")
    print(tokenizer.decode(first_target[0].tolist())[:200])
    print(f"\nVerification: target should be input shifted by 1 token")
    print(f"Input[1:5]:  {first_input[0][1:5].tolist()}")
    print(f"Target[0:4]: {first_target[0][0:4].tolist()}")
    print(f"Match: {(first_input[0][1:5] == first_target[0][0:4]).all().item()}")
    print("="*60 + "\n")
    
    for epoch in range(start_epoch, config["num_epochs"]):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{config['num_epochs']}")
        print(f"{'='*60}")
        
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            # Check for shutdown signal
            global _shutdown_requested
            if _shutdown_requested:
                print(f"\n⚠️  Graceful shutdown at step {global_step}")
                print("  Saving emergency checkpoint...")
                try:
                    last_loss = loss.item()
                except:
                    last_loss = 0.0
                save_checkpoint(
                    model, optimizer, epoch, global_step,
                    last_loss, best_val_loss, best_val_loss, checkpoint_dir,
                    batches_per_epoch
                )
                print("  ✓ Emergency checkpoint saved. Exiting cleanly.")
                return
            
            # Skip batches if resuming mid-epoch
            if epoch == start_epoch and batch_idx < skip_batches:
                continue
            
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
            global_step += 1
            
            # Print training progress
            if global_step % 10 == 0:
                print(f"Step {global_step:05d} | Batch {batch_idx:04d} | Loss: {loss.item():.4f}")
            
            # Evaluation
            if global_step % config["eval_freq"] == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, config["eval_iters"]
                )
                
                print(f"\n{'─'*60}")
                print(f"📊 EVALUATION at Step {global_step}")
                print(f"   Train Loss: {train_loss:.4f}")
                print(f"   Val Loss:   {val_loss:.4f}")
                
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
                    print(f"  Best model saved: {best_model_path} (val_loss: {val_loss:.4f})")
            
            # Save periodic checkpoint
            if global_step % config["save_every_n_iterations"] == 0:
                save_checkpoint(
                    model, optimizer, epoch, global_step,
                    loss.item(), best_val_loss, best_val_loss, checkpoint_dir,
                    batches_per_epoch
                )
    
    # Save final checkpoint
    save_checkpoint(
        model, optimizer, epoch, global_step,
        loss.item(), best_val_loss, best_val_loss, checkpoint_dir,
        batches_per_epoch
    )
    print("[OK] Final checkpoint saved")
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")


# ============================================================================
# DATA PREPARATION  
# ============================================================================

CACHE_DIR = Path(__file__).parent / "cache"


def get_cache_path(num_samples: int) -> Path:
    """Genera el path del cache basado en num_samples"""
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"synth_en_samples_{num_samples}.parquet"


def prepare_data_from_synth(max_length=256, train_split=0.9, num_samples=10000):
    """
    Load and prepare SYNTH dataset from HuggingFace.
    Usa cache en formato Parquet para evitar reprocesar.
    """
    cache_path = get_cache_path(num_samples)
    
    # Check cache
    if cache_path.exists():
        print(f"[CACHE] Cargando desde {cache_path}...")
        import pandas as pd
        df = pd.read_parquet(cache_path)
        texts = df['text'].tolist()
        print(f"[OK] {len(texts):,} textos cargados desde cache")
    else:
        print("Loading SYNTH dataset from HuggingFace...")
        print(f"Target samples: {num_samples:,}\n")
        
        dataset = load_dataset("PleIAs/SYNTH", split="train", streaming=True)
        texts = []
        print("Collecting texts from dataset...")
        
        global _shutdown_requested
        
        for i, item in enumerate(dataset):
            if _shutdown_requested:
                print(f"\n  [WARN] Shutdown requested. Stopping data collection.")
                print(f"  Collected {len(texts):,} samples before shutdown.")
                if len(texts) < 1000:
                    print("  ERROR: Not enough samples collected. Exiting.")
                    sys.exit(0)
                print("  Continuing with partial dataset...\n")
                break
            
            if len(texts) >= num_samples:
                break
            
            if item.get('language') != 'en':
                continue
            
            query = item.get('query', '')
            reasoning = item.get('synthetic_reasoning', '')
            answer = item.get('synthetic_answer', '')
            
            if query and answer:
                if reasoning:
                    text = f"Q: {query}\n\nReasoning:\n{reasoning}\n\nA: {answer}"
                else:
                    text = f"Q: {query}\n\nA: {answer}"
                texts.append(text)
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1} examples, collected {len(texts)} English samples...")
        
        print(f"\n[OK] Collected {len(texts):,} texts")
        
        # Save to cache
        print(f"[CACHE] Guardando en {cache_path}...")
        import pandas as pd
        df = pd.DataFrame({'text': texts})
        df.to_parquet(cache_path, index=False)
        print(f"[OK] Cache guardado")
    
    # DEBUG: Show sample
    print("\n" + "="*60)
    print("DEBUG: Sample texts from dataset")
    print("="*60)
    for i, sample_text in enumerate(texts[:2]):
        print(f"\n--- Example {i+1} (first 500 chars) ---")
        print(sample_text[:500])
        print("...")
    print("="*60 + "\n")
    
    # Combine all texts with separator
    full_text = "\n\n---\n\n".join(texts)
    print(f"Total characters: {len(full_text):,}")
    
    # Split into train and validation
    split_idx = int(len(full_text) * train_split)
    train_text = full_text[:split_idx]
    val_text = full_text[split_idx:]
    
    print(f"Train characters: {len(train_text):,}")
    print(f"Val characters: {len(val_text):,}")
    
    return train_text, val_text


def create_dataloaders(train_text, val_text, batch_size, max_length):
    """Create train and validation dataloaders"""    
    train_loader = create_dataloader_v1(
        train_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    
    val_loader = create_dataloader_v1(
        val_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    return train_loader, val_loader


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("GPT-2 PRE-TRAINING FROM SCRATCH")
    print("="*60 + "\n")
    
    device = TRAINING_CONFIG["device"]
    
    # ========================================
    # STEP 1: Setup paths and validate checkpoint FIRST (before slow dataset loading)
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
            print("✓ Checkpoint validated successfully. Proceeding with dataset loading...\n")
        except Exception as e:
            print(f"\n❌ CHECKPOINT ERROR: {e}")
            print("\nAborting before dataset loading to save time.")
            print("Please fix checkpoint_to_resume or set it to None.\n")
            sys.exit(1)
    
    # Store validated checkpoint in config for later use
    TRAINING_CONFIG["_validated_checkpoint"] = validated_checkpoint
    
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
    # STEP 3: Prepare data (slow operation)
    # ========================================
    train_text, val_text = prepare_data_from_synth(
        max_length=TRAINING_CONFIG["max_length"],
        num_samples=TRAINING_CONFIG["num_samples"]
    )
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_text, 
        val_text,
        batch_size=TRAINING_CONFIG["batch_size"],
        max_length=TRAINING_CONFIG["max_length"]
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"]
    )
    
    # Train
    train_model(model, train_loader, val_loader, optimizer, TRAINING_CONFIG)


if __name__ == "__main__":
    main()

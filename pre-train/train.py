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
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arch.gpt_model import GPTModel
from arch.config import GPT_CONFIG_124M
from utils import tokenizer, text_to_token_ids, token_ids_to_text, generate_text_simple, create_dataloader_v1 

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    # Data
    "num_samples": 50_000,      # ~3-5M tokens, enough for meaningful pre-training
    "max_length": 512,          # Longer context for Q&A reasoning
    
    # Training
    "batch_size": 8,            # Increase if VRAM allows (RTX 3090: 16, RTX 4090: 32)
    "num_epochs": 2,            # 2 passes over data is typical for pre-training
    "learning_rate": 3e-4,      # Standard for GPT-2 size models
    "weight_decay": 0.1,        # Regularization
    "gradient_clip": 1.0,       # Prevent exploding gradients
    "warmup_steps": 500,        # Linear warmup (TODO: implement scheduler)
    
    # Evaluation & Checkpoints
    "eval_freq": 500,           # Evaluate every N steps
    "eval_iters": 20,           # More batches for stable eval metrics
    "save_every_n_iterations": 2000,  # Save checkpoint every N steps
    
    # Resume training
    "checkpoint_to_use": None,  # Path to checkpoint to resume from (e.g., "checkpoints/checkpoint_step_2000.pt")
    
    # Hardware
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

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

def load_checkpoint(checkpoint_path: Path, model: GPTModel, optimizer: torch.optim.Optimizer, device: torch.device):
    """
    Load a checkpoint and restore training state.
    Returns: (global_step, start_epoch, best_val_loss)
    """
    print("\n" + "="*60)
    print("RESUMING FROM CHECKPOINT")
    print("="*60)
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint: dict = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    global_step = checkpoint['global_step']
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    train_loss = checkpoint.get('train_loss', 'N/A')
    val_loss = checkpoint.get('val_loss', 'N/A')
    
    print(f"\n  Checkpoint info:")
    print(f"  - Global step: {global_step:,}")
    print(f"  - Epoch: {epoch + 1}")
    print(f"  - Train loss at save: {train_loss}")
    print(f"  - Val loss at save: {val_loss}")
    print(f"  - Best val loss so far: {best_val_loss:.4f}")
    print(f"\n  Resuming training from step {global_step + 1}...")
    print("="*60 + "\n")
    
    return global_step, epoch, best_val_loss


def save_checkpoint(model: GPTModel, optimizer: torch.optim.Optimizer, epoch: int, global_step: int, 
                    train_loss: float, val_loss: float, best_val_loss: float, checkpoint_dir: Path):
    """Save a training checkpoint"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{global_step}.pt"
    
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
    }, checkpoint_path)
    
    print(f"\n  Checkpoint saved: {checkpoint_path}")
    
    # Keep only last 3 checkpoints to save disk space
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))
    if len(checkpoints) > 3:
        for old_ckpt in checkpoints[:-3]:
            old_ckpt.unlink()
            print(f"  Removed old checkpoint: {old_ckpt.name}")


def train_model(model: GPTModel, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer, config: dict):
    """Main training loop"""
    device: torch.device = config["device"]
    model.to(device)
    model.train()
    
    # Setup checkpoint directory
    checkpoint_dir = Path("checkpoints")
    
    # Initialize training state
    global_step = 0
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Load checkpoint if specified
    checkpoint_path = config.get("checkpoint_to_use")
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            global_step, start_epoch, best_val_loss = load_checkpoint(
                checkpoint_path, model, optimizer, device
            )
        else:
            print(f"\n  WARNING: Checkpoint not found: {checkpoint_path}")
            print(f"  Starting training from scratch...\n")
    
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
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, 'best_model.pt')
                    print(f"  Best model saved (val_loss: {val_loss:.4f})")
            
            # Save periodic checkpoint
            if global_step % config["save_every_n_iterations"] == 0:
                save_checkpoint(
                    model, optimizer, epoch, global_step,
                    loss.item(), best_val_loss, best_val_loss, checkpoint_dir
                )
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data_from_synth(max_length=256, train_split=0.9, num_samples=10000):
    """
    Load and prepare SYNTH dataset from HuggingFace
    
    SYNTH dataset fields:
    - synth_id: unique ID
    - language: language code (en, fr, es, etc.)
    - query: the question/prompt
    - synthetic_reasoning: step-by-step reasoning
    - synthetic_answer: the generated answer
    """
    print("Loading SYNTH dataset from HuggingFace...")
    print(f"Target samples: {num_samples:,}\n")
    
    # Load dataset in streaming mode
    dataset = load_dataset("PleIAs/SYNTH", split="train", streaming=True)
    
    # Collect texts
    texts = []
    print("Collecting texts from dataset...")

    assert isinstance(dataset, IterableDataset)
    
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        
        # Filter for English only (optional - remove if you want multilingual)
        if item.get('language') != 'en':
            continue
        
        # Combine query + reasoning + answer for rich training data
        # This gives the model both the question and the reasoning process
        query = item.get('query', '')
        reasoning = item.get('synthetic_reasoning', '')
        answer = item.get('synthetic_answer', '')
        
        # Format: Q: [query]\nReasoning: [reasoning]\nA: [answer]
        if query and answer:
            if reasoning:
                # Include reasoning for better learning
                text = f"Q: {query}\n\nReasoning:\n{reasoning}\n\nA: {answer}"
            else:
                # Simple Q&A format
                text = f"Q: {query}\n\nA: {answer}"
            
            texts.append(text)
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1} examples, collected {len(texts)} English samples...")
    
    print(f"\n✓ Collected {len(texts):,} texts")
    
    # DEBUG: Show sample of collected texts
    print("\n" + "="*60)
    print("DEBUG: Sample texts from dataset")
    print("="*60)
    for i, sample_text in enumerate(texts[:2]):  # Show first 2 examples
        print(f"\n--- Example {i+1} (first 500 chars) ---")
        print(sample_text[:500])
        print("...")
    print("="*60 + "\n")
    
    # Combine all texts with separator
    full_text = "\n\n---\n\n".join(texts)
    print(f"Total characters: {len(full_text):,}")
    print(f"Total tokens: {len(tokenizer.encode(full_text)):,}")
    
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
    
    # Initialize model
    print("Initializing model...")
    model = GPTModel(GPT_CONFIG_124M)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Model config: {GPT_CONFIG_124M}\n")
    
    # Prepare data
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

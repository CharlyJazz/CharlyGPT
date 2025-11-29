# GPT-2 From Scratch

Pre-training GPT-2 124M from scratch using the SYNTH dataset.

## Project Structure

```
.
├── arch/                       # Model architecture
│   ├── attention.py            # Multi-head causal self-attention
│   ├── config.py               # Model configuration (GPT_CONFIG_124M)
│   ├── feed_forward.py         # Feed-forward network
│   ├── gelu.py                 # GELU activation
│   ├── gpt_model.py            # Main GPTModel class
│   ├── layer_norm.py           # Layer normalization
│   └── transformer_block.py    # Transformer block
├── pre-train/                  # Pre-training scripts
│   ├── train.py                # Main training loop
│   └── utils.py                # Tokenizer, dataset, generation utils
├── post-train/                 # Fine-tuning (TODO)
│   └── train.py
└── checkpoints/                # Saved model checkpoints (auto-created)
```

## Quick Start

### Pre-training

```bash
cd pre-train
python train.py
```

### Configuration

Edit `pre-train/train.py`:

```python
TRAINING_CONFIG = {
    # Data
    "num_samples": 50_000,          # Samples from SYNTH dataset
    "max_length": 512,              # Context length for training
    
    # Training
    "batch_size": 8,                # Adjust based on GPU VRAM
    "num_epochs": 2,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "gradient_clip": 1.0,
    
    # Checkpoints
    "eval_freq": 500,               # Evaluate every N steps
    "save_every_n_iterations": 2000,# Save checkpoint every N steps
    "checkpoint_to_use": None,      # Path to resume training
}
```

### Resume Training

To resume from a checkpoint:

```python
"checkpoint_to_use": "checkpoints/checkpoint_step_4000.pt",
```

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Parameters | ~124M |
| Vocab Size | 50,257 (GPT-2 BPE) |
| Context Length | 1,024 tokens |
| Embedding Dim | 768 |
| Layers | 12 |
| Attention Heads | 12 |
| Dropout | 0.1 |

## Dataset

### Pre-training
- **SYNTH**: https://huggingface.co/datasets/PleIAs/SYNTH
  - High-quality synthetic Q&A with reasoning
  - Format: `Q: [query]\n\nReasoning:\n[reasoning]\n\nA: [answer]`

### Post-training
- TODO: Instruction fine-tuning datasets

## Requirements

```bash
pip install torch tiktoken datasets
```

## Features

- Checkpoint saving/resuming
- Real-time loss monitoring
- Text generation samples during training
- Gradient clipping
- Train/validation split
- Auto-cleanup of old checkpoints (keeps last 3)

## Based On

"Build a Large Language Model From Scratch" - Sebastian Raschka

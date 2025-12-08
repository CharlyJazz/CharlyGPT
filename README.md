# GPT-2 From Scratch

Pre-training GPT-2 124M from scratch using the SYNTH dataset with **streaming data loading**.

## Project Structure

```
.
├── arch/                           # Model architecture
│   ├── attention.py                # Multi-head causal self-attention
│   ├── config.py                   # Model configuration (GPT_CONFIG_124M)
│   ├── feed_forward.py             # Feed-forward network
│   ├── gelu.py                     # GELU activation
│   ├── gpt_model.py                # Main GPTModel class
│   ├── layer_norm.py               # Layer normalization
│   └── transformer_block.py        # Transformer block
├── pre-train/                      # Pre-training scripts
│   ├── train.py                    # Main training loop
│   ├── utils.py                    # Tokenizer and generation utilities
│   ├── mlflow_viewer.py            # MLflow checkpoint evaluation
│   ├── experiments/                # YAML experiment configs
│   │   └── SmallGPT2-Samples2M.yaml
│   ├── dataset/                    # Streaming dataset module
│   │   ├── __init__.py
│   │   ├── config.py               # StreamingConfig dataclass
│   │   ├── tokenizer_utils.py      # Tokenizer utilities
│   │   ├── streaming_dataset.py    # StreamingGPTDataset (IterableDataset)
│   │   ├── dataloader_factory.py   # create_streaming_dataloaders()
│   │   ├── checkpoint_manager.py   # Dataset state checkpointing
│   │   └── test_streaming.py       # Integration tests
│   └── mlruns/                     # MLflow tracking (auto-created)
├── post-train/                     # Fine-tuning (TODO)
│   └── train.py
└── E:\GPT_SANDBOX_STORAGE\         # External storage for checkpoints
    └── <experiment_name>\
        └── checkpoints\
            ├── checkpoint_step_*.pt
            └── best_model.pt
```

## Quick Start

### 1. Configure Experiment

Create/edit a YAML config in `pre-train/experiments/`:

```yaml
# experiments/SmallGPT2-Samples2M.yaml
experiment_name: Traditional Small GPT 2 - Samples 2M

data:
  num_samples: 2000000    # Max samples to process (null = unlimited)
  max_length: 512         # Sequence length (context size)
  buffer_size: 10000      # Shuffle buffer size (higher = better randomization)
  seed: 42                # Random seed for reproducibility
  train_ratio: 0.9        # 90% train, 10% validation

training:
  batch_size: 8
  num_epochs: 2
  learning_rate: 0.0003
  weight_decay: 0.1
  gradient_clip: 1.0
  warmup_steps: 500

evaluation:
  eval_freq: 500
  eval_iters: 20
  save_every_n_iterations: 1000

storage:
  base_folder: E:\GPT_SANDBOX_STORAGE
  checkpoint_to_resume: null  # or "checkpoint_step_50000.pt"

hardware:
  device: cuda
```

### 2. Run Training

```bash
cd pre-train
python train.py
```

The script will:
1. Load config from `experiments/SmallGPT2-Samples2M.yaml`
2. Create streaming dataloaders (instant, no pre-loading)
3. Stream data from HuggingFace on-the-fly (~1GB RAM)
4. Save checkpoints to `E:\GPT_SANDBOX_STORAGE\<experiment_name>\checkpoints\`

### 3. Resume Training

Edit the YAML:

```yaml
storage:
  checkpoint_to_resume: checkpoint_step_50000.pt
```

Then run `python train.py` again.

### 4. Evaluate Checkpoints with MLflow

```bash
python mlflow_viewer.py --config experiments/SmallGPT2-Samples2M.yaml --questions 5
```

This will:
- Find all checkpoints in the experiment folder
- Evaluate each with test questions from SYNTH
- Log results to MLflow (idempotent - won't repeat)

View results:

```bash
python -m mlflow ui
# Open http://localhost:5000
```

## Streaming Dataset

Training uses **memory-efficient streaming** instead of loading all data into RAM:

```
============================================================
CREATING STREAMING DATALOADERS
============================================================
  Dataset: PleIAs/SYNTH
  Max samples: 2,000,000
  Max length: 512
  Batch size: 8
  Buffer size: 10000
[OK] Streaming dataloaders created (no data loaded yet)
```

### How it works

1. **No pre-loading**: Data streams from HuggingFace on-demand
2. **On-the-fly tokenization**: Each sample tokenized as needed
3. **Shuffle buffer**: Approximate shuffling with configurable buffer size
4. **Low memory**: ~1GB RAM regardless of dataset size

### Memory comparison

| Method | RAM Usage |
|--------|-----------|
| Old (load all) | ~15-20 GB |
| **Streaming** | **~1 GB** |

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

- **SYNTH**: https://huggingface.co/datasets/PleIAs/SYNTH
  - High-quality synthetic Q&A with reasoning
  - Format: `Q: [query]\n\nReasoning:\n[reasoning]\n\nA: [answer]`
  - Filtered to English only

## Requirements

```bash
pip install torch tiktoken datasets mlflow

# Optional: for mid-epoch checkpointing
pip install torchdata>=0.8.0
```

## Features

- **Streaming dataset** - Memory-efficient, ~1GB RAM regardless of dataset size
- **On-the-fly tokenization** - No pre-processing step needed
- **YAML-based configuration** - Easy experiment management
- **Checkpoint management** - Auto-save, resume, keep last 3
- **MLflow integration** - Track and compare checkpoints
- **Idempotent evaluation** - Won't duplicate MLflow runs
- **Graceful shutdown** - Ctrl+C saves emergency checkpoint
- **Instant startup** - No waiting for data loading

## Running Tests

```bash
cd pre-train/dataset
python test_streaming.py
```

Tests use the real SYNTH dataset (no mocks).

## Based On

"Build a Large Language Model From Scratch" - Sebastian Raschka

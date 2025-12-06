# GPT-2 From Scratch

Pre-training GPT-2 124M from scratch using the SYNTH dataset.

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
│   ├── mlflow_viewer.py            # MLflow checkpoint evaluation
│   ├── experiments/                # YAML experiment configs
│   │   └── SmallGPT2-Samples2M.yaml
│   ├── cache/                      # Dataset cache (auto-created)
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
  num_samples: 2000000
  max_length: 512

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
2. Cache dataset to `cache/synth_en_samples_2000000.parquet` (first run only)
3. Save checkpoints to `E:\GPT_SANDBOX_STORAGE\<experiment_name>\checkpoints\`
4. Generate runtime info (parameters, iterations) in the YAML

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

## Dataset Caching

First run processes the SYNTH dataset and caches it:

```
Loading SYNTH dataset from HuggingFace...
  Processed 1000 examples, collected 810 English samples...
  ...
[OK] Collected 2,000,000 texts
[CACHE] Guardando en cache/synth_en_samples_2000000.parquet...
```

Subsequent runs load instantly:

```
[CACHE] Cargando desde cache/synth_en_samples_2000000.parquet...
[OK] 2,000,000 textos cargados desde cache
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

- **SYNTH**: https://huggingface.co/datasets/PleIAs/SYNTH
  - High-quality synthetic Q&A with reasoning
  - Format: `Q: [query]\n\nReasoning:\n[reasoning]\n\nA: [answer]`
  - Filtered to English only

## Requirements

```bash
pip install torch tiktoken datasets mlflow pandas pyarrow
```

## Features

- **YAML-based configuration** - Easy experiment management
- **Dataset caching** - Parquet format for fast loading
- **Checkpoint management** - Auto-save, resume, keep last 3
- **MLflow integration** - Track and compare checkpoints
- **Idempotent evaluation** - Won't duplicate MLflow runs
- **Graceful shutdown** - Ctrl+C saves emergency checkpoint
- **Runtime info** - Auto-calculates iterations, tokens, parameters

## Based On

"Build a Large Language Model From Scratch" - Sebastian Raschka

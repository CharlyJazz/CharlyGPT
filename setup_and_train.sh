#!/bin/bash

################################################################################
# RunPod Training Setup & Execution Script
# 
# This script automates the complete setup and training process for RunPod:
# 1. Detects project directory automatically
# 2. Verifies Python and CUDA availability
# 3. Checks and installs missing dependencies
# 4. Configures environment variables
# 5. Executes training with proper error handling
#
# Usage:
#   chmod +x setup_and_train.sh
#   ./setup_and_train.sh [experiment_yaml]
#
# Example:
#   ./setup_and_train.sh pre-train/experiments/Experiment2-ChatML-Optimizations-RunPod.yaml
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "================================================================================"
    echo "$1"
    echo "================================================================================"
    echo ""
}

################################################################################
# STAGE 1: Environment Detection
################################################################################

print_header "STAGE 1: ENVIRONMENT DETECTION"

# Detect project root (script should be in project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

log_info "Project root: $PROJECT_ROOT"
log_info "Current user: $(whoami)"
log_info "Current directory: $(pwd)"

# Verify Python
if ! command -v python &> /dev/null; then
    log_error "Python not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
log_success "Python version: $PYTHON_VERSION"

# Verify CUDA availability
if command -v nvidia-smi &> /dev/null; then
    log_success "NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    log_warning "No NVIDIA GPU detected. Training will use CPU (very slow!)"
fi

################################################################################
# STAGE 2: Dependency Verification
################################################################################

print_header "STAGE 2: DEPENDENCY VERIFICATION"

check_python_package() {
    local package=$1
    if python -c "import $package" 2>/dev/null; then
        log_success "$package is installed"
        return 0
    else
        log_warning "$package is NOT installed"
        return 1
    fi
}

# List of required packages (import names)
REQUIRED_PACKAGES=(
    "torch"
    "tiktoken"
    "yaml"
    "datasets"
    "mlflow"
)

MISSING_PACKAGES=()

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! check_python_package "$pkg"; then
        MISSING_PACKAGES+=("$pkg")
    fi
done

################################################################################
# STAGE 3: Dependency Installation
################################################################################

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    print_header "STAGE 3: INSTALLING MISSING DEPENDENCIES"
    
    log_info "Missing packages: ${MISSING_PACKAGES[*]}"
    
    # Install PyTorch with CUDA support
    if [[ " ${MISSING_PACKAGES[*]} " =~ " torch " ]]; then
        log_info "Installing PyTorch with CUDA 12.1 support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        log_success "PyTorch installed"
    fi
    
    # Install other dependencies
    log_info "Installing project dependencies..."
    
    # Check if requirements.txt exists
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        log_info "Found requirements.txt, installing..."
        pip install --ignore-installed -r "$PROJECT_ROOT/requirements.txt" || {
            log_warning "Installation with requirements.txt failed, trying individual packages..."
            pip install --ignore-installed tiktoken pyyaml datasets huggingface_hub mlflow
        }
    else
        log_warning "No requirements.txt found, installing packages individually..."
        pip install --ignore-installed tiktoken pyyaml datasets huggingface_hub mlflow
    fi
    
    # Optional: Install torchdata for StatefulDataLoader (better checkpointing)
    log_info "Installing optional dependencies..."
    pip install torchdata>=0.8.0 || log_warning "Could not install torchdata (optional)"
    
    log_success "All dependencies installed"
else
    print_header "STAGE 3: DEPENDENCY CHECK"
    log_success "All required dependencies are already installed"
fi

################################################################################
# STAGE 4: Configuration
################################################################################

print_header "STAGE 4: CONFIGURATION"

# Determine experiment YAML file
if [ $# -eq 0 ]; then
    # No argument provided, use default RunPod config
    EXPERIMENT_FILE="$PROJECT_ROOT/pre-train/experiments/Experiment2-ChatML-Optimizations-RunPod.yaml"
    log_info "No experiment file specified, using default: Experiment2-ChatML-Optimizations-RunPod.yaml"
else
    # Use provided argument
    EXPERIMENT_FILE="$1"
    log_info "Using experiment file: $EXPERIMENT_FILE"
fi

# Convert to absolute path if relative
if [[ "$EXPERIMENT_FILE" != /* ]]; then
    EXPERIMENT_FILE="$PROJECT_ROOT/$EXPERIMENT_FILE"
fi

# Verify experiment file exists
if [ ! -f "$EXPERIMENT_FILE" ]; then
    log_error "Experiment file not found: $EXPERIMENT_FILE"
    log_info "Available experiments:"
    ls -1 "$PROJECT_ROOT/pre-train/experiments/"*.yaml 2>/dev/null || log_warning "No YAML files found in experiments/"
    exit 1
fi

log_success "Experiment file found: $EXPERIMENT_FILE"

# Set environment variable
export EXPERIMENT_FILE="$EXPERIMENT_FILE"
log_success "Environment variable EXPERIMENT_FILE set"

# Display experiment configuration
log_info "Experiment configuration:"
echo "---"
head -n 20 "$EXPERIMENT_FILE"
echo "..."

################################################################################
# STAGE 4.5: Checkpoint Verification & Download
################################################################################

print_header "STAGE 4.5: CHECKPOINT VERIFICATION"

# Parse YAML to extract checkpoint_to_resume and base_folder
CHECKPOINT_TO_RESUME=$(python -c "
import yaml
with open('$EXPERIMENT_FILE', 'r') as f:
    config = yaml.safe_load(f)
    ckpt = config.get('storage', {}).get('checkpoint_to_resume')
    print(ckpt if ckpt else '')
" 2>/dev/null || echo "")

BASE_FOLDER=$(python -c "
import yaml
with open('$EXPERIMENT_FILE', 'r') as f:
    config = yaml.safe_load(f)
    base = config.get('storage', {}).get('base_folder', '.')
    print(base)
" 2>/dev/null || echo ".")

EXPERIMENT_NAME=$(python -c "
import yaml
with open('$EXPERIMENT_FILE', 'r') as f:
    config = yaml.safe_load(f)
    name = config.get('experiment_name', 'default_experiment')
    print(name)
" 2>/dev/null || echo "default_experiment")

if [ -n "$CHECKPOINT_TO_RESUME" ] && [ "$CHECKPOINT_TO_RESUME" != "null" ]; then
    CHECKPOINT_PATH="$BASE_FOLDER/$EXPERIMENT_NAME/checkpoints/$CHECKPOINT_TO_RESUME"
    
    log_info "Checkpoint configured: $CHECKPOINT_TO_RESUME"
    log_info "Expected path: $CHECKPOINT_PATH"
    
    if [ -f "$CHECKPOINT_PATH" ]; then
        CHECKPOINT_SIZE=$(du -h "$CHECKPOINT_PATH" | cut -f1)
        log_success "Checkpoint found! Size: $CHECKPOINT_SIZE"
    else
        log_warning "Checkpoint NOT found in volume!"
        log_info "Checkpoint path: $CHECKPOINT_PATH"
        echo ""
        log_error "CHECKPOINT MISSING - Training cannot resume!"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📤 UPLOAD CHECKPOINT FROM YOUR LOCAL MACHINE (Windows/Mac/Linux)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "Option 1: Using runpodctl (RECOMMENDED - no pod needed)"
        echo "  From your LOCAL terminal (not this pod):"
        echo ""
        echo "  # Install runpodctl (one-time setup)"
        echo "  # Windows: Download from https://github.com/runpod/runpodctl/releases"
        echo "  # Linux/Mac: wget https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64"
        echo ""
        echo "  # Configure API key (one-time)"
        echo "  runpodctl config --apiKey YOUR_RUNPOD_API_KEY"
        echo ""
        echo "  # Upload checkpoint"
        echo "  runpodctl send <local_checkpoint_path> <volume_id>:/$EXPERIMENT_NAME/checkpoints/"
        echo ""
        echo "  Example:"
        echo "  runpodctl send \"C:\\Users\\...\\checkpoint_step_171849.pt\" mrt3b0u1x6:/$EXPERIMENT_NAME/checkpoints/"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "Option 2: Using SCP (requires this pod running)"
        echo "  From your LOCAL terminal:"
        echo ""
        echo "  # Get SSH connection info from RunPod UI → Your Pod → Connect → SSH"
        echo "  # Then run:"
        echo "  scp -P <port> <local_checkpoint_path> root@<ip>:$CHECKPOINT_PATH"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "Option 3: Start training from scratch (no checkpoint)"
        echo "  Edit YAML to set checkpoint_to_resume: null"
        echo "  sed -i 's/checkpoint_to_resume: .*/checkpoint_to_resume: null/' $EXPERIMENT_FILE"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "📖 See UPLOAD_CHECKPOINT.md for detailed step-by-step instructions"
        echo ""
        
        # Ask user what to do
        read -p "Do you want to continue WITHOUT checkpoint (start from scratch)? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_error "Aborting. Please upload checkpoint from your local machine."
            echo ""
            echo "After uploading, run this script again:"
            echo "  ./setup_and_train.sh"
            echo ""
            exit 1
        else
            log_warning "Continuing WITHOUT checkpoint - training will start from scratch"
        fi
    fi
else
    log_info "No checkpoint configured - training will start from scratch"
fi

################################################################################
# STAGE 5: Pre-flight Checks
################################################################################

print_header "STAGE 5: PRE-FLIGHT CHECKS"

# Verify training script exists
TRAIN_SCRIPT="$PROJECT_ROOT/pre-train/train.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    log_error "Training script not found: $TRAIN_SCRIPT"
    exit 1
fi
log_success "Training script found: $TRAIN_SCRIPT"

# Check disk space
AVAILABLE_SPACE=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
log_info "Available disk space: $AVAILABLE_SPACE"

# Verify CUDA is accessible from Python
if python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    CUDA_DEVICES=$(python -c "import torch; print(torch.cuda.device_count())")
    CUDA_DEVICE_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    log_success "CUDA available: $CUDA_DEVICES device(s) - $CUDA_DEVICE_NAME"
else
    log_warning "CUDA not available in PyTorch. Training will be slow!"
fi

################################################################################
# STAGE 6: Training Execution
################################################################################

print_header "STAGE 6: STARTING TRAINING"

log_info "Changing to project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

log_info "Executing training script..."
echo ""
echo "================================================================================"
echo "TRAINING OUTPUT (press Ctrl+C to stop gracefully)"
echo "================================================================================"
echo ""

# Execute training with proper error handling
if python "$TRAIN_SCRIPT"; then
    echo ""
    print_header "TRAINING COMPLETED SUCCESSFULLY"
    log_success "Training finished without errors"
    exit 0
else
    EXIT_CODE=$?
    echo ""
    print_header "TRAINING FAILED"
    log_error "Training exited with code: $EXIT_CODE"
    exit $EXIT_CODE
fi

#!/bin/bash
# ============================================================================
# RunPod Initial Setup Script
# ============================================================================
# This script clones the repository and starts the training process.
# Run this once when you first connect to a new RunPod instance.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/CharlyJazz/CharlyGPT/experiment-2/runpod_init.sh | bash
#
# Or manually:
#   chmod +x runpod_init.sh
#   ./runpod_init.sh
# ============================================================================

set -e  # Exit on error

echo ""
echo "================================================================================"
echo "RunPod Initial Setup - CharlyGPT Training"
echo "================================================================================"
echo ""

# Navigate to workspace
cd /workspace

# Check if repo already exists
if [ -d "CharlyGPT" ]; then
    echo "[INFO] Repository already exists, pulling latest changes..."
    cd CharlyGPT
    git pull
    git checkout experiment-2
else
    echo "[INFO] Cloning repository..."
    git clone https://github.com/CharlyJazz/CharlyGPT.git
    cd CharlyGPT
    git checkout experiment-2
fi

# Make script executable
echo "[INFO] Making setup script executable..."
chmod +x setup_and_train.sh

echo ""
echo "================================================================================"
echo "Setup Complete!"
echo "================================================================================"
echo ""
echo "To start training, run:"
echo "  ./setup_and_train.sh pre-train/experiments/Experiment2-ChatML-Optimizations-RunPod.yaml"
echo ""
echo "Or for H100 GPU:"
echo "  ./setup_and_train.sh pre-train/experiments/Experiment2-ChatML-Optimizations-H100.yaml"
echo ""
echo "================================================================================"
echo ""

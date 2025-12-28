#!/bin/bash
# End-to-End Experiment Script
#
# This script runs a full training + evaluation cycle using the
# bank_account_embeddings package.
#
# Prerequisites:
#   1. Clone the repo and install: pip install -e .
#   2. Set up config: cp config/default.yaml config/local.yaml
#   3. Prepare data (tensors, vocabularies, emerging_flags.csv)
#
# Usage:
#   ./scripts/run_experiment.sh [EXP_ID]
#   ./scripts/run_experiment.sh repro_14k
#
# The script expects data in $DATA_DIR (from config or default ./data/)

set -e

# Experiment ID (default: test)
EXP_ID="${1:-test}"

# Directories (override via environment variables if needed)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"
EXP_DIR="$RESULTS_DIR/exp_${EXP_ID}"

# Model hyperparameters (match exp_14k by default)
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-4}"
NUM_HEADS="${NUM_HEADS:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-0.0001}"

echo "========================================================"
echo "Experiment: $EXP_ID"
echo "Data Dir:   $DATA_DIR"
echo "Output Dir: $EXP_DIR"
echo "Model:      hidden_dim=$HIDDEN_DIM, layers=$NUM_LAYERS, heads=$NUM_HEADS"
echo "Training:   epochs=$EPOCHS, batch_size=$BATCH_SIZE, lr=$LR"
echo "Started:    $(date)"
echo "========================================================"

# Create output directories
mkdir -p "$EXP_DIR/model"
mkdir -p "$EXP_DIR/eval"

# Validate required files
TENSOR_FILE="${TENSOR_FILE:-$DATA_DIR/pretrain_tensors.pt}"
EMERGING_CSV="${EMERGING_CSV:-$DATA_DIR/emerging_flags.csv}"
VOCAB_DIR="${VOCAB_DIR:-$DATA_DIR/vocabularies}"

if [ ! -f "$TENSOR_FILE" ]; then
    echo "ERROR: Tensor file not found: $TENSOR_FILE"
    echo "Please prepare tensors using: python -m hierarchical.data.preprocess_tensors"
    exit 1
fi

if [ ! -f "$EMERGING_CSV" ]; then
    echo "ERROR: Emerging flags CSV not found: $EMERGING_CSV"
    exit 1
fi

# Copy vocabularies to model dir (required by training script)
if [ -d "$VOCAB_DIR" ]; then
    echo "[1/3] Copying vocabularies..."
    mkdir -p "$EXP_DIR/model/vocabularies"
    cp -r "$VOCAB_DIR"/* "$EXP_DIR/model/vocabularies/"
else
    echo "WARNING: Vocabulary directory not found: $VOCAB_DIR"
    echo "Training will build vocabularies from tensors if possible."
fi

# Phase 1: Training
echo ""
echo "[2/3] Training Model..."
echo "----------------------------------------------------------------"

python -m hierarchical.training.pretrain \
    --preprocessed_file "$TENSOR_FILE" \
    --output_dir "$EXP_DIR/model" \
    --hidden_dim "$HIDDEN_DIM" \
    --num_layers "$NUM_LAYERS" \
    --num_heads "$NUM_HEADS" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --num_workers 4 \
    --use_balance \
    --use_amp

# Phase 2: Evaluation
echo ""
echo "[3/3] Running Evaluation..."
echo "----------------------------------------------------------------"

# Find the best model checkpoint
MODEL_CKPT="$EXP_DIR/model/model_best.pth"
if [ ! -f "$MODEL_CKPT" ]; then
    MODEL_CKPT="$EXP_DIR/model/best_model.pt"
fi

python -m hierarchical.evaluation.unified_eval_optimized \
    --tensors_path "$TENSOR_FILE" \
    --emerging_csv "$EMERGING_CSV" \
    --model_checkpoint "$MODEL_CKPT" \
    --output_dir "$EXP_DIR/eval" \
    --hidden_dim "$HIDDEN_DIM" \
    --T_values 7 \
    --num_seeds 3

echo ""
echo "========================================================"
echo "Experiment $EXP_ID Complete!"
echo "Results:    $EXP_DIR/eval/"
echo "Finished:   $(date)"
echo "========================================================"

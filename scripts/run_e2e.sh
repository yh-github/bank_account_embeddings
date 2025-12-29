#!/bin/bash
# End-to-End Experiment from Raw Data
#
# This script runs a FULL training + evaluation cycle starting from
# raw transaction CSV files. It verifies the entire pipeline:
#   1. Preprocess: CSV → Tensors + Vocabularies
#   2. Train: Tensors → Model
#   3. Evaluate: Model + Tensors + Labels → Metrics
#
# Prerequisites:
#   1. Clone the repo and install: pip install -e .
#   2. Have raw data available (transactions.csv, accounts.csv, emerging_flags.csv)
#
# Usage:
#   DATA_DIR=/path/to/data ./scripts/run_e2e.sh [EXP_ID]
#
# Required files in DATA_DIR:
#   - transactions.csv: Raw transaction data
#   - accounts.csv: Account metadata with balances
#   - emerging_flags.csv: Evaluation labels

set -e

# Experiment ID (default: e2e_test)
EXP_ID="${1:-e2e_test}"

# Directories
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"
EXP_DIR="$RESULTS_DIR/exp_${EXP_ID}"

# Model hyperparameters
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-4}"
NUM_HEADS="${NUM_HEADS:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-0.0001}"

# Preprocessing parameters
MIN_DAYS="${MIN_DAYS:-10}"
MAX_DAYS="${MAX_DAYS:-180}"

echo "========================================================"
echo "End-to-End Experiment: $EXP_ID"
echo "========================================================"
echo "Data Dir:   $DATA_DIR"
echo "Output Dir: $EXP_DIR"
echo "Model:      hidden_dim=$HIDDEN_DIM, layers=$NUM_LAYERS, heads=$NUM_HEADS"
echo "Training:   epochs=$EPOCHS, batch_size=$BATCH_SIZE, lr=$LR"
echo "Started:    $(date)"
echo "========================================================"

# Create directories
mkdir -p "$EXP_DIR/model/vocabularies"
mkdir -p "$EXP_DIR/eval"
mkdir -p "$EXP_DIR/data"

# Validate required files
TXN_FILE="$DATA_DIR/transactions.csv"
ACC_FILE="$DATA_DIR/accounts.csv"
EMERGING_CSV="$DATA_DIR/emerging_flags.csv"

if [ ! -f "$TXN_FILE" ]; then
    echo "ERROR: transactions.csv not found: $TXN_FILE"
    exit 1
fi

if [ ! -f "$ACC_FILE" ]; then
    echo "ERROR: accounts.csv not found: $ACC_FILE"
    exit 1
fi

if [ ! -f "$EMERGING_CSV" ]; then
    echo "ERROR: emerging_flags.csv not found: $EMERGING_CSV"
    exit 1
fi

# ================================================================
# Phase 1: Preprocessing (CSV → Tensors)
# ================================================================
echo ""
echo "[1/3] Preprocessing: CSV → Tensors"
echo "----------------------------------------------------------------"

TENSOR_FILE="$EXP_DIR/data/pretrain_tensors.pt"

if [ -f "$TENSOR_FILE" ]; then
    echo "Skipping preprocessing: $TENSOR_FILE exists"
else
    python -m hierarchical.data.preprocess_tensors \
        --mode train \
        --data_file "$TXN_FILE" \
        --account_file "$ACC_FILE" \
        --vocab_dir "$EXP_DIR/model/vocabularies" \
        --output_dir "$EXP_DIR/data" \
        --min_days "$MIN_DAYS" \
        --max_days "$MAX_DAYS"
fi

# Verify preprocessing output
if [ ! -f "$TENSOR_FILE" ]; then
    echo "ERROR: Preprocessing failed - no tensors created"
    exit 1
fi

echo "Tensors: $(ls -lh "$TENSOR_FILE" | awk '{print $5}')"
echo "Vocabularies: $(ls "$EXP_DIR/model/vocabularies/")"

# ================================================================
# Phase 2: Training
# ================================================================
echo ""
echo "[2/3] Training Model"
echo "----------------------------------------------------------------"

MODEL_CKPT="$EXP_DIR/model/model_best.pth"

if [ -f "$MODEL_CKPT" ]; then
    echo "Skipping training: $MODEL_CKPT exists"
else
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
fi

# Verify training output
if [ ! -f "$MODEL_CKPT" ]; then
    echo "ERROR: Training failed - no model checkpoint created"
    exit 1
fi

echo "Model: $(ls -lh "$MODEL_CKPT" | awk '{print $5}')"

# ================================================================
# Phase 3: Evaluation
# ================================================================
echo ""
echo "[3/3] Running Evaluation"
echo "----------------------------------------------------------------"

python -m hierarchical.evaluation.evaluate \
    --tensors_path "$TENSOR_FILE" \
    --emerging_csv "$EMERGING_CSV" \
    --model_checkpoint "$MODEL_CKPT" \
    --output_dir "$EXP_DIR/eval" \
    --hidden_dim "$HIDDEN_DIM" \
    --T_values 7 \
    --seeds 3

# ================================================================
# Summary
# ================================================================
echo ""
echo "========================================================"
echo "Experiment $EXP_ID Complete!"
echo "========================================================"
echo "Finished:   $(date)"
echo ""
echo "Outputs:"
echo "  Tensors: $TENSOR_FILE"
echo "  Model:   $MODEL_CKPT"
echo "  Eval:    $EXP_DIR/eval/"
echo ""

# Show key metrics if available
if [ -f "$EXP_DIR/eval/all_lift_metrics.csv" ]; then
    echo "Key Metrics:"
    head -10 "$EXP_DIR/eval/all_lift_metrics.csv"
fi

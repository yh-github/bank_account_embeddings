# Bank Account Embeddings

A self-supervised framework for learning account-level embeddings from banking transaction sequences using a hierarchical transformer architecture.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd hierarchical-embeddings

# Install the package
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all]"
```

## Configuration

Before running training or data preparation, set up your configuration:

```bash
# Copy the default config and update paths
cp config/default.yaml config/local.yaml

# Edit config/local.yaml with your data paths
# Then set the environment variable
export EMBEDDER_CONFIG=config/local.yaml
```

Key configuration options in `config/local.yaml`:
- `paths.data_dir`: Root directory containing bank subdirectories with transaction data
- `paths.output_dir`: Where to save processed data and results
- `banks`: List of bank subdirectory names to process

## Architecture

```
Transaction → TransactionEncoder → Day Embedding → DayEncoder → Account Embedding
                   ↓                     ↓                           ↓
              (per-txn)            (per-day)                   (per-account)
```

### Model Components

| Component | Description |
|-----------|-------------|
| **TransactionEncoder** | Encodes individual transactions using categorical embeddings (category group, sub-category, counter-party) and numerical features (amount, date) |
| **DayEncoder** | Aggregates daily transactions into a single vector using transformer + mean pooling. Handles positive (credits) and negative (debits) streams separately |
| **AccountEncoder** | Processes sequences of day embeddings with positional and calendar encodings to produce a final account embedding |

### Training

Pre-training uses **contrastive learning** (SimCLR-style) with the InfoNCE loss. Two views of the same account are created through data augmentation (temporal cropping, random day dropping).

## Usage

### Quick Start

```python
from hierarchical.models import AccountEncoder, DayEncoder, TransactionEncoder
from hierarchical.data import HierarchicalDataset, collate_hierarchical, build_vocabularies
from hierarchical.training import contrastive_loss

# 1. Build vocabularies from your transaction data
cat_grp, cat_sub, counter_party = build_vocabularies(transactions_df)

# 2. Initialize encoder hierarchy  
txn_enc = TransactionEncoder(
    num_categories_group=len(cat_grp),
    num_categories_sub=len(cat_sub),
    num_counter_parties=len(counter_party),
    embedding_dim=128
)
day_enc = DayEncoder(txn_enc, hidden_dim=128)
model = AccountEncoder(day_enc, hidden_dim=128)

# 3. Create dataset and dataloader
dataset = HierarchicalDataset(transactions_df, cat_grp, cat_sub, counter_party)
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_hierarchical)

# 4. Training loop
for batch in loader:
    embeddings = model(batch)  # [B, 128]
    # ... compute loss and backprop
```

### Data Preparation

```bash
# Consolidate data from multiple banks
python -m scripts.data_prep.consolidate_data --config config/local.yaml

# Generate financial flags (labels)
python -m scripts.data_prep.generate_labels --config config/local.yaml
```

### Pre-training from CLI

```bash
python -m hierarchical.training.pretrain \
    --data_file output/transactions.csv \
    --output_dir models/model_v1 \
    --epochs 10 \
    --batch_size 32 \
    --hidden_dim 128
```

### Evaluation

```bash
python -m hierarchical.evaluation.evaluate \
    --model_path models/model_v1/model_best.pth \
    --data_path path/to/eval_data \
    --output_dir results/eval
```

## Project Structure

```
hierarchical-embeddings/
├── config/
│   └── default.yaml       # Configuration template
├── src/hierarchical/
│   ├── config.py          # Configuration loader
│   ├── data/              # Data loading, vocabularies, balance features
│   ├── models/            # TransactionEncoder, DayEncoder, AccountEncoder
│   ├── training/          # Pre-training script, loss functions
│   └── evaluation/        # Downstream evaluation pipeline
├── scripts/
│   └── data_prep/         # Data preparation scripts
├── tests/                 # Unit tests
├── pyproject.toml         # Package definition
└── requirements.txt       # Dependencies
```

## Running Tests

```bash
# Using pytest (recommended)
python -m pytest tests/ -v

# Or using the test runner
python run_tests.py
```

## Key Features

- **Hierarchical Structure**: Transaction → Day → Account for multi-scale representation
- **Contrastive Pre-training**: Self-supervised learning without labels
- **Flexible Architecture**: Optional SetTransformer, balance features, counter-party embeddings
- **Data Augmentation**: Temporal cropping, random day dropping for contrastive views
- **Configuration System**: YAML-based config for all paths and parameters


import argparse
import logging
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from hierarchical.models.account import AccountEncoder
from hierarchical.models.day import DayEncoder
from hierarchical.models.transaction import TransactionEncoder
from hierarchical.data.preloaded_dataset import PreloadedDataset
from hierarchical.data.dataset import collate_hierarchical
from hierarchical.data.vocab import load_vocabularies
from hierarchical.training.utils import recursive_to_device
from hierarchical.evaluation.unified_eval_optimized import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Note: load_model is imported from unified_eval_optimized which was patched to accept 'args'
# But we need to make sure we pass 'args' to it.

def get_embeddings(model, dataset, device, batch_size=128, num_workers=4):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_hierarchical, num_workers=num_workers
    )
    
    embeddings = []
    account_ids = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating Embeddings"):
            batch_dev = recursive_to_device(batch, device)
            emb = model(batch_dev)
            embeddings.append(emb.cpu().numpy())
            account_ids.extend(batch['account_id'])
            
    return np.concatenate(embeddings, axis=0), account_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', required=True)
    parser.add_argument('--vocab_dir', required=True)
    parser.add_argument('--preprocessed_file', required=True)
    parser.add_argument('--accounts_csv', required=True, help="Path to accounts.csv with 'type' column")
    parser.add_argument('--output_dir', required=True)
    
    # Model Args (Must match training)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--txn_dim', type=int, help='Override transaction dimension')
    parser.add_argument('--day_dim', type=int, help='Override day dimension')
    parser.add_argument('--account_dim', type=int, help='Override account dimension')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--use_balance', action='store_true', default=True) # Defaulting to True for Exp14 Prime
    parser.add_argument('--no_counter_party', action='store_true')
    parser.add_argument('--model_type', default='hierarchical')
    
    parser.add_argument('--max_samples', type=int, default=10000, help="Max accounts to visualize")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Model
    # Pass 'args' so that load_model can pick up the dimension overrides
    model = load_model(args.model_checkpoint, device, args.hidden_dim, args.model_type, args)
    
    # 2. Load Data
    logger.info(f"Loading data from {args.preprocessed_file}")
    all_data = torch.load(args.preprocessed_file, weights_only=False)
    
    # Filter to max samples if needed
    if len(all_data) > args.max_samples:
        np.random.seed(42)
        indices = np.random.choice(len(all_data), args.max_samples, replace=False)
        data_sample = [all_data[i] for i in indices]
    else:
        data_sample = all_data
        
    dataset = PreloadedDataset(data_sample, augment=False)
    
    # 3. Get Embeddings
    embs, ids = get_embeddings(model, dataset, device)
    
    # 4. Merge with Account Types
    df_acc = pd.read_csv(args.accounts_csv)
    # Deduplicate accounts by ID
    df_acc = df_acc.drop_duplicates(subset=['id'])
    
    df_emb = pd.DataFrame({'id': ids})
    
    # Merge
    merged = df_emb.merge(df_acc[['id', 'type']], on='id', how='left')
    
    # Filter out missing types
    valid_mask = merged['type'].notna()
    embs_valid = embs[valid_mask]
    labels_valid = merged.loc[valid_mask, 'type'].values
    ids_valid = merged.loc[valid_mask, 'id'].values
    
    logger.info(f"Analyze {len(embs_valid)} accounts with known types.")
    logger.info(f"Type Counts:\n{pd.Series(labels_valid).value_counts()}")
    
    # 5. Dimensionality Reduction
    logger.info("Running PCA...")
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(embs_valid)
    
    # Try UMAP if available
    umap_res = None
    try:
        import umap
        logger.info("Running UMAP...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
        umap_res = reducer.fit_transform(embs_valid)
    except ImportError:
        logger.warning("UMAP not installed, skipping.")
        
    # 6. Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=pca_res[:,0], y=pca_res[:,1], hue=labels_valid, alpha=0.6, s=15)
    plt.title("PCA of Account Embeddings by Type")
    plt.savefig(os.path.join(args.output_dir, "pca_account_types.png"))
    plt.close()
    
    if umap_res is not None:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=umap_res[:,0], y=umap_res[:,1], hue=labels_valid, alpha=0.6, s=15)
        plt.title("UMAP of Account Embeddings by Type")
        plt.savefig(os.path.join(args.output_dir, "umap_account_types.png"))
        plt.close()
        
    # 7. Metrics
    # Filter to main types for cleaner silhouette
    main_types = ['Checking', 'Savings', 'Credit Card', 'Loan'] # Adjust based on data
    mask = np.isin(labels_valid, main_types)
    if mask.sum() > 100:
        score = silhouette_score(embs_valid[mask], labels_valid[mask], metric='cosine')
        logger.info(f"Silhouette Score (Main Types): {score:.4f}")
        
        with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
            f.write(f"Silhouette Score: {score:.4f}\n")
            f.write(f"Counts:\n{pd.Series(labels_valid).value_counts().to_string()}\n")
    else:
        logger.warning("Not enough samples of main types for robust silhouette score.")

if __name__ == '__main__':
    main()

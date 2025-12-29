# Deployment Guide for Bank Account Embeddings (Final)

> [!IMPORTANT]
> This guide replaces all previous deployment instructions. **Do not use** the `embeddings_poc` virtual environment or directories.

## 1. Remote Machine Details
- **IP**: `10.218.0.9`
- **User**: `azureuser`
- **Project Directory**: `~/projects/bank_account_embeddings`
- **Virtual Environment**: `~/projects/bank_account_embeddings/venv`

## 2. Initial Setup (One-Time)

### Connect to Remote
```bash
ssh -i ~/.ssh/A100-data-science-vm_key.pem azureuser@10.218.0.9
```

### Clean Install
Run the following commands on the **remote machine** to set up the project from scratch:

```bash
# 1. Create directory (if not exists) and navigate
mkdir -p ~/projects/bank_account_embeddings
cd ~/projects/bank_account_embeddings

# 2. Pull latest code (assuming git is initialized)
# If starting fresh: git clone <repo_url> .
git pull origin main

# 3. Create a FRESH virtual environment
python3 -m venv venv

# 4. Activate and Install Dependencies
source venv/bin/activate
pip install --upgrade pip

# CRITICAL: Install in editable mode to register the 'hierarchical' package
pip install -e .
```

## 3. Running Experiments

### End-to-End Test
Use the provided script. It handles preprocessing, training, and evaluation.

```bash
# Activate the dedicated venv
cd ~/projects/bank_account_embeddings
source venv/bin/activate

# Run the E2E script (use nohup for long runs)
nohup bash scripts/run_e2e.sh > e2e.log 2>&1 &
```

### Monitoring
```bash
tail -f e2e.log
```

## 4. Updates
When pulling new code:
```bash
git pull
# If requirements change:
pip install -r requirements.txt
```

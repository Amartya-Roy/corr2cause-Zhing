#!/usr/bin/env python3
"""
generate_and_push.py

End-to-end pipeline:
  1. Generate causal NLI data for n=7..50 using data_gen_large.py
  2. Post-process into HF dataset format with columns:
       - input:  "Premise: ...\nHypothesis: ..."
       - label:  0 or 1  (1=entailment, 0=non-entailment)
       - num_variables: int
       - template: str (e.g. "has_collider", "parent", etc.)
  3. Save as CSV + Parquet
  4. Push to Hugging Face Hub
"""

import os
import sys
import json
import re
import time
from collections import defaultdict

import pandas as pd

# Import generation logic from data_gen_large.py (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_gen_large import generate_data_for_n, num_samples2splits

# ============================================================================
# Configuration
# ============================================================================
MIN_NODES = 7
MAX_NODES = 26
NUM_DAGS = 100
NUM_MEC_SAMPLES = 20
EDGE_PROB = None          # auto
SEED = 0
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
HF_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hf_dataset')
HF_REPO_ID = "Amartya77/Extended_Corr2Cause"


def extract_template_from_id(sample_id: str) -> str:
    """Extract causal_relation (template) from the structured id string."""
    # id format: num_nodes=4__mec_id=0__node_i=1__node_j=2__causal_relation=has_collider__prob=0.50
    match = re.search(r'causal_relation=([^_]+(?:_[^_]+)*?)__prob=', sample_id)
    if match:
        return match.group(1).replace('_', ' ')  # e.g. "non parent ancestor" -> keep as template name
    # fallback: try without __prob
    match = re.search(r'causal_relation=(.+?)(?:__|$)', sample_id)
    if match:
        return match.group(1).replace('_', ' ')
    return "unknown"


def extract_num_variables_from_id(sample_id: str) -> int:
    """Extract num_nodes from the structured id string."""
    match = re.search(r'num_nodes=(\d+)', sample_id)
    if match:
        return int(match.group(1))
    return -1


def nli_to_hf_row(sample: dict) -> dict:
    """Convert a single NLI sample to the HF dataset format."""
    input_text = f"Premise: {sample['premise']}\nHypothesis: {sample['hypothesis']}"
    label = 1 if sample['relation'] == 'entailment' else 0
    num_variables = extract_num_variables_from_id(sample['id'])
    template = extract_template_from_id(sample['id'])
    return {
        'input': input_text,
        'label': label,
        'num_variables': num_variables,
        'template': template,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(HF_DATASET_DIR, exist_ok=True)

    total_t0 = time.time()
    all_rows = []

    # ---------------------------------------------------------------
    # Step 1: Generate NLI data for each n
    # ---------------------------------------------------------------
    for n in range(MIN_NODES, MAX_NODES + 1):
        json_file = os.path.join(OUTPUT_DIR, f'causalnli_{n}nodes.json')

        # Check if already generated (resume support)
        if os.path.exists(json_file):
            print(f"[Info] Loading existing {json_file}")
            with open(json_file) as f:
                nli_data = json.load(f)
        else:
            nli_data = generate_data_for_n(
                n,
                num_dags=NUM_DAGS,
                num_mec_samples=NUM_MEC_SAMPLES,
                edge_prob=EDGE_PROB,
                seed=SEED,
            )
            # Save raw JSON
            indent = 2 if n <= 15 else None
            with open(json_file, 'w') as f:
                json.dump(nli_data, f, indent=indent)
            size_mb = os.path.getsize(json_file) / 1e6
            print(f"[Info] Saved {len(nli_data)} samples ({size_mb:.1f} MB) -> {json_file}")

        # Convert to HF rows
        for sample in nli_data:
            all_rows.append(nli_to_hf_row(sample))

    total_elapsed = time.time() - total_t0
    print(f"\n[Info] Generation complete: {len(all_rows)} total rows in {total_elapsed:.1f}s")

    # ---------------------------------------------------------------
    # Step 2: Build DataFrame and save
    # ---------------------------------------------------------------
    df = pd.DataFrame(all_rows)
    print(f"\n[Info] Dataset shape: {df.shape}")
    print(f"[Info] Label distribution:\n{df['label'].value_counts()}")
    print(f"[Info] Template distribution:\n{df['template'].value_counts()}")
    print(f"[Info] Num variables range: {df['num_variables'].min()} - {df['num_variables'].max()}")
    print(f"\n[Info] Sample rows:")
    print(df.head(3).to_string(max_colwidth=120))

    # Save CSV
    csv_path = os.path.join(HF_DATASET_DIR, 'data.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n[Info] Saved CSV -> {csv_path} ({os.path.getsize(csv_path)/1e6:.1f} MB)")

    # Save Parquet (smaller, faster for HF)
    parquet_path = os.path.join(HF_DATASET_DIR, 'data.parquet')
    df.to_parquet(parquet_path, index=False)
    print(f"[Info] Saved Parquet -> {parquet_path} ({os.path.getsize(parquet_path)/1e6:.1f} MB)")

    # Also save train/dev/test splits
    split2rows = defaultdict(list)
    for n in range(MIN_NODES, MAX_NODES + 1):
        n_rows = [r for r in all_rows if r['num_variables'] == n]
        if not n_rows:
            continue
        splits = num_samples2splits(len(n_rows))
        idx = 0
        for split_name, size in splits.items():
            split2rows[split_name].extend(n_rows[idx:idx + size])
            idx += size

    for split_name, rows in split2rows.items():
        split_df = pd.DataFrame(rows)
        split_csv = os.path.join(HF_DATASET_DIR, f'{split_name}.csv')
        split_parquet = os.path.join(HF_DATASET_DIR, f'{split_name}.parquet')
        split_df.to_csv(split_csv, index=False)
        split_df.to_parquet(split_parquet, index=False)
        print(f"[Info] {split_name:>5}: {len(rows):>8} rows -> {split_csv}")

    # ---------------------------------------------------------------
    # Step 3: Create dataset card (README.md)
    # ---------------------------------------------------------------
    readme_content = f"""---
license: mit
task_categories:
  - text-classification
language:
  - en
tags:
  - causal-inference
  - NLI
  - causal-reasoning
size_categories:
  - 100K<n<1M
---

# Extended Corr2Cause Dataset

Extended version of the Corr2Cause dataset with causal NLI examples
for graphs with {MIN_NODES} to {MAX_NODES} variables.

## Dataset Structure

| Column | Type | Description |
|--------|------|-------------|
| `input` | str | "Premise: ...\\nHypothesis: ..." |
| `label` | int | 0 = non-entailment, 1 = entailment |
| `num_variables` | int | Number of variables in the causal system ({MIN_NODES}-{MAX_NODES}) |
| `template` | str | Causal relation type (e.g., "has collider", "parent", etc.) |

## Templates

The following causal relation templates are used:
- **parent**: X directly causes Y
- **non parent ancestor**: X causes something else which causes Y
- **child**: Y directly causes X
- **non child descendant**: Y is a cause for X, but not a direct one
- **has collider**: There exists at least one common effect of X and Y
- **has confounder**: There exists at least one common cause of X and Y

## Statistics

- Total examples: {len(all_rows):,}
- Label 0 (non-entailment): {df['label'].value_counts().get(0, 0):,}
- Label 1 (entailment): {df['label'].value_counts().get(1, 0):,}
- Variable range: {MIN_NODES} to {MAX_NODES}

## Usage

```python
from datasets import load_dataset
ds = load_dataset("Amartya77/Extended_Corr2Cause")
```

## Splits

- train: {len(split2rows.get('train', [])):,}
- dev: {len(split2rows.get('dev', [])):,}
- test: {len(split2rows.get('test', [])):,}
"""

    readme_path = os.path.join(HF_DATASET_DIR, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"[Info] Saved dataset card -> {readme_path}")

    # ---------------------------------------------------------------
    # Step 4: Push to Hugging Face Hub
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Pushing to Hugging Face Hub")
    print(f"{'=' * 60}")

    from huggingface_hub import login, upload_folder

    # Login (will prompt for token if not already logged in)
    login()

    # Push the dataset folder
    upload_folder(
        folder_path=HF_DATASET_DIR,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )

    print(f"\n[Info] Dataset pushed to https://huggingface.co/datasets/{HF_REPO_ID}")
    print("[Done]")


if __name__ == '__main__':
    main()

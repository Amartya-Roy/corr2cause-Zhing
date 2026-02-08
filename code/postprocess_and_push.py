#!/usr/bin/env python3
"""
postprocess_and_push.py

Memory-efficient post-processing and HF push.
Processes one JSON file at a time to avoid OOM.
"""

import os
import sys
import json
import re
import time
from collections import defaultdict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_gen_large import num_samples2splits

# ============================================================================
# Configuration
# ============================================================================
MIN_NODES = 7
MAX_NODES = 26
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
HF_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hf_dataset')
HF_REPO_ID = "Amartya77/Extended_Corr2Cause"


def extract_template_from_id(sample_id: str) -> str:
    match = re.search(r'causal_relation=([^_]+(?:_[^_]+)*?)__prob=', sample_id)
    if match:
        return match.group(1).replace('_', ' ')
    match = re.search(r'causal_relation=(.+?)(?:__|$)', sample_id)
    if match:
        return match.group(1).replace('_', ' ')
    return "unknown"


def extract_num_variables_from_id(sample_id: str) -> int:
    match = re.search(r'num_nodes=(\d+)', sample_id)
    if match:
        return int(match.group(1))
    return -1


def nli_to_hf_row(sample: dict) -> dict:
    premise = sample.get('premise', '')
    hypothesis = sample.get('hypothesis', '')
    input_text = f"Premise: {premise}\nHypothesis: {hypothesis}"
    label_str = sample.get('label', 'contradiction')
    label = 1 if label_str == 'entailment' else 0
    sample_id = sample.get('id', '')
    num_vars = extract_num_variables_from_id(sample_id)
    template = extract_template_from_id(sample_id)
    return {
        'input': input_text,
        'label': label,
        'num_variables': num_vars,
        'template': template,
    }


def process_one_file(json_path: str) -> pd.DataFrame:
    """Load one JSON file, convert to DataFrame, and free memory."""
    print(f"  Loading {os.path.basename(json_path)} ...", end=" ", flush=True)
    with open(json_path, 'r') as f:
        data = json.load(f)
    rows = [nli_to_hf_row(s) for s in data]
    del data  # free raw JSON memory
    df = pd.DataFrame(rows)
    del rows
    print(f"{len(df)} rows")
    return df


def main():
    os.makedirs(HF_DATASET_DIR, exist_ok=True)
    # Subdirectory for per-n parquet files
    data_subdir = os.path.join(HF_DATASET_DIR, 'data')
    os.makedirs(data_subdir, exist_ok=True)

    total_rows = 0
    label_counts = defaultdict(int)
    template_counts = defaultdict(int)
    split_writers = {}  # split_name -> list of DataFrames (written per-n)
    split_counts = defaultdict(int)

    print(f"\n{'='*60}")
    print(f"  Post-processing n={MIN_NODES}..{MAX_NODES}")
    print(f"{'='*60}\n")

    # Process one file at a time
    for idx, n in enumerate(range(MIN_NODES, MAX_NODES + 1)):
        json_file = os.path.join(DATA_DIR, f'causalnli_{n}nodes.json')
        if not os.path.exists(json_file):
            print(f"  [SKIP] {json_file} not found")
            continue

        df = process_one_file(json_file)
        total_rows += len(df)

        # Accumulate stats
        for lbl, cnt in df['label'].value_counts().items():
            label_counts[lbl] += cnt
        for tmpl, cnt in df['template'].value_counts().items():
            template_counts[tmpl] += cnt

        # Save per-n parquet shard
        shard_name = f"data-{idx:05d}-of-{MAX_NODES - MIN_NODES + 1:05d}.parquet"
        shard_path = os.path.join(data_subdir, shard_name)
        df.to_parquet(shard_path, index=False)
        print(f"    -> {shard_name} ({os.path.getsize(shard_path)/1e6:.1f} MB)")

        # Compute splits for this n
        splits = num_samples2splits(len(df))
        idx_start = 0
        for split_name, size in splits.items():
            split_df = df.iloc[idx_start:idx_start + size]
            idx_start += size
            split_counts[split_name] += len(split_df)

            # Append to split parquet file
            split_parquet = os.path.join(HF_DATASET_DIR, f'{split_name}.parquet')
            if os.path.exists(split_parquet):
                # Read existing, concatenate, rewrite
                existing = pd.read_parquet(split_parquet)
                combined = pd.concat([existing, split_df], ignore_index=True)
                combined.to_parquet(split_parquet, index=False)
                del existing, combined
            else:
                split_df.to_parquet(split_parquet, index=False)

        del df  # free memory

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Label distribution:")
    for lbl in sorted(label_counts):
        print(f"    {lbl}: {label_counts[lbl]:,}")
    print(f"  Templates:")
    for tmpl, cnt in sorted(template_counts.items(), key=lambda x: -x[1]):
        print(f"    {tmpl}: {cnt:,}")
    print(f"  Splits:")
    for split_name, cnt in split_counts.items():
        print(f"    {split_name}: {cnt:,}")

    # ---------------------------------------------------------------
    # Create dataset card (README.md)
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
  - 1M<n<10M
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

- **parent**: X directly causes Y
- **non parent ancestor**: X causes something else which causes Y
- **child**: Y directly causes X
- **non child descendant**: Y is a cause for X, but not a direct one
- **has collider**: There exists at least one common effect of X and Y
- **has confounder**: There exists at least one common cause of X and Y

## Statistics

- Total examples: {total_rows:,}
- Label 0 (non-entailment): {label_counts.get(0, 0):,}
- Label 1 (entailment): {label_counts.get(1, 0):,}
- Variable range: {MIN_NODES} to {MAX_NODES}

## Usage

```python
from datasets import load_dataset
ds = load_dataset("Amartya77/Extended_Corr2Cause")
```

## Splits

- train: {split_counts.get('train', 0):,}
- dev: {split_counts.get('dev', 0):,}
- test: {split_counts.get('test', 0):,}
"""

    readme_path = os.path.join(HF_DATASET_DIR, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"\n[Info] Saved dataset card -> {readme_path}")

    # ---------------------------------------------------------------
    # Push to Hugging Face Hub
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Pushing to Hugging Face Hub")
    print(f"{'='*60}")

    from huggingface_hub import login, upload_folder

    login()

    upload_folder(
        folder_path=HF_DATASET_DIR,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )

    print(f"\n[Info] Dataset pushed to https://huggingface.co/datasets/{HF_REPO_ID}")
    print("[Done]")


if __name__ == '__main__':
    main()

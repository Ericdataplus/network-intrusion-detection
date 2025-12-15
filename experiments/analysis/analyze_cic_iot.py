"""
CIC IoT 2023 DATASET ANALYSIS
=============================
Analyze the CIC IoT 2023 dataset (33 attack types from 105 real IoT devices)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import joblib

print("=" * 80)
print("CIC IoT 2023 DATASET ANALYSIS")
print("33 Attack Types | 105 Real IoT Devices | 2023")
print("=" * 80)

# Find all CSV parts
iot_path = Path.home() / ".cache/kagglehub/datasets/madhavmalhotra/unb-cic-iot-dataset/versions/1/wataiData/csv/CICIoT2023"
csv_files = sorted(glob.glob(str(iot_path / "*.csv")))

print(f"\nFound {len(csv_files)} CSV files")

# Sample a few files to understand structure
print("\nLoading sample files...")
sample_dfs = []
for i, f in enumerate(csv_files[:5]):  # First 5 files
    df = pd.read_csv(f, nrows=10000)  # 10K rows each = 50K sample
    sample_dfs.append(df)
    if i == 0:
        print(f"   Columns: {len(df.columns)}")
        print(f"   Sample columns: {list(df.columns[:10])}")

sample_df = pd.concat(sample_dfs, ignore_index=True)
print(f"\nSample size: {len(sample_df):,} records")

# Check for label column
label_col = None
for col in sample_df.columns:
    if 'label' in col.lower() or 'attack' in col.lower() or 'class' in col.lower():
        print(f"\nFound label column: {col}")
        print(f"   Unique values: {sample_df[col].nunique()}")
        print(f"   Value counts:\n{sample_df[col].value_counts().head(15)}")
        label_col = col
        break

# Dataset stats
print("\n" + "=" * 60)
print("DATASET STATISTICS")
print("=" * 60)
print(f"   Total columns: {len(sample_df.columns)}")
print(f"   Sample rows: {len(sample_df):,}")
print(f"   Numeric columns: {sample_df.select_dtypes(include=[np.number]).shape[1]}")

# Estimate total size
avg_rows_per_file = 500000  # Estimate
total_estimated = len(csv_files) * avg_rows_per_file
print(f"   Estimated total records: ~{total_estimated:,}")

# Attack types if found
if label_col:
    attack_types = sample_df[label_col].unique()
    print(f"\n   Attack types found: {len(attack_types)}")
    print(f"   Types: {list(attack_types[:20])}")

# Save summary
summary = {
    'dataset': 'CIC IoT 2023',
    'source': 'Canadian Institute for Cybersecurity',
    'files': len(csv_files),
    'columns': len(sample_df.columns),
    'estimated_records': total_estimated,
    'label_column': label_col,
    'attack_types': list(attack_types) if label_col else [],
    'column_names': list(sample_df.columns)
}

joblib.dump(summary, 'models/cic_iot_summary.joblib')
print("\nSaved CIC IoT 2023 summary to models/cic_iot_summary.joblib")

print("\n" + "=" * 80)
print("CIC IoT 2023 ANALYSIS COMPLETE")
print("=" * 80)

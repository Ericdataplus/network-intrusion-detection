"""
UNSW-NB15 Network Intrusion Detection Dataset - Initial Exploration
====================================================================
Dataset: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

This script explores the UNSW-NB15 dataset to understand:
- Data structure and features
- Attack type distribution
- Class balance
- Feature statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load files
print("=" * 70)
print("UNSW-NB15 NETWORK INTRUSION DETECTION DATASET")
print("=" * 70)

# Load training and testing sets
print("\n[1] Loading datasets...")
train_df = pd.read_csv('training_set.csv')
test_df = pd.read_csv('testing_set.csv')

print(f"   Training set: {len(train_df):,} records")
print(f"   Testing set:  {len(test_df):,} records")
print(f"   Total:        {len(train_df) + len(test_df):,} records")

# Basic info
print("\n[2] Dataset Structure")
print("-" * 40)
print(f"   Columns: {len(train_df.columns)}")
print(f"\n   Column names:")
for i, col in enumerate(train_df.columns):
    print(f"      {i+1:2d}. {col}")

# Data types
print("\n[3] Data Types")
print("-" * 40)
print(train_df.dtypes.value_counts())

# Target variable analysis
print("\n[4] Target Variable: 'label' (Binary)")
print("-" * 40)
if 'label' in train_df.columns:
    label_counts = train_df['label'].value_counts()
    print(f"   0 (Normal):  {label_counts.get(0, 0):,} ({100*label_counts.get(0, 0)/len(train_df):.1f}%)")
    print(f"   1 (Attack):  {label_counts.get(1, 0):,} ({100*label_counts.get(1, 0)/len(train_df):.1f}%)")

# Attack categories
print("\n[5] Attack Categories: 'attack_cat'")
print("-" * 40)
if 'attack_cat' in train_df.columns:
    attack_counts = train_df['attack_cat'].value_counts()
    print(f"\n   Found {len(attack_counts)} categories:\n")
    for cat, count in attack_counts.items():
        pct = 100 * count / len(train_df)
        bar = '#' * int(pct)
        print(f"   {cat:15s}: {count:6,} ({pct:5.1f}%) {bar}")

# Numeric features summary
print("\n[6] Numeric Features Summary")
print("-" * 40)
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
print(f"   Numeric columns: {len(numeric_cols)}")
print("\n   Sample statistics (first 10 numeric features):")
print(train_df[numeric_cols[:10]].describe().round(2).to_string())

# Missing values
print("\n[7] Missing Values")
print("-" * 40)
missing = train_df.isnull().sum()
missing_pct = 100 * missing / len(train_df)
if missing.sum() > 0:
    missing_df = pd.DataFrame({'Missing': missing, 'Percent': missing_pct})
    missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
    print(missing_df.to_string())
else:
    print("   No missing values found!")

# Categorical columns
print("\n[8] Categorical Columns")
print("-" * 40)
cat_cols = train_df.select_dtypes(include=['object']).columns
print(f"   Categorical columns: {len(cat_cols)}")
for col in cat_cols:
    print(f"\n   {col}: {train_df[col].nunique()} unique values")
    print(f"   Top 5: {train_df[col].value_counts().head().to_dict()}")

# Key insights
print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)

if 'attack_cat' in train_df.columns and 'label' in train_df.columns:
    normal_count = (train_df['attack_cat'] == 'Normal').sum()
    attack_count = len(train_df) - normal_count
    
    print(f"""
1. DATASET SIZE:
   - Training: {len(train_df):,} records
   - Testing: {len(test_df):,} records
   - Features: {len(train_df.columns) - 2} (excluding id and label)

2. CLASS DISTRIBUTION:
   - Normal traffic: {normal_count:,} ({100*normal_count/len(train_df):.1f}%)
   - Attack traffic: {attack_count:,} ({100*attack_count/len(train_df):.1f}%)

3. ATTACK TYPES (9 categories):
   - Exploits, DoS, Fuzzers, Reconnaissance, Generic
   - Backdoors, Analysis, Shellcode, Worms

4. ML OPPORTUNITIES:
   - Binary classification (Normal vs Attack)
   - Multi-class classification (9 attack types + Normal)
   - Feature importance analysis
   - Anomaly detection
""")

print("=" * 70)
print("Exploration complete! Ready for analysis.")
print("=" * 70)

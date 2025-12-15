"""
Quick extraction of comprehensive analysis results
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

print("=" * 70)
print("COMPREHENSIVE RESULTS EXTRACTION")
print("=" * 70)

# Load the existing results
results = joblib.load('models/results_summary.joblib')

# Check files in models folder
print("\nModels saved:")
for f in Path('models').glob('*'):
    print(f"   {f.name}: {f.stat().st_size / 1024:.1f} KB")

# Load main datasets for stats
train_df = pd.read_csv('training_set.csv')
test_df = pd.read_csv('testing_set.csv')
unsw_df = pd.concat([train_df, test_df], ignore_index=True)
unsw_df['attack_cat'] = unsw_df['attack_cat'].str.strip()

# Load NSL-KDD 
nsl_train = pd.read_csv('data/nsl_kdd/Train_data.csv')
nsl_test = pd.read_csv('data/nsl_kdd/Test_data.csv')
nsl_df = pd.concat([nsl_train, nsl_test], ignore_index=True)

# Count CICIDS files
cicids_files = list(Path('data/cicids2017').glob('*.csv'))

print(f"\n{'='*70}")
print("DATASET SUMMARY")
print(f"{'='*70}")
print(f"UNSW-NB15: {len(unsw_df):,} records")
print(f"NSL-KDD: {len(nsl_df):,} records") 
print(f"CICIDS2017: {len(cicids_files)} files (800MB+)")

# Extract existing binary results
print(f"\n{'='*70}")
print("EXISTING MODEL RESULTS (Binary Classification)")
print(f"{'='*70}")

for model, metrics in results['binary_results'].items():
    print(f"{model}: {metrics['accuracy']*100:.2f}% accuracy, {metrics['roc_auc']:.4f} ROC-AUC")

# Attack distribution
print(f"\n{'='*70}")
print("ATTACK DISTRIBUTION (UNSW-NB15)")
print(f"{'='*70}")
for cat, count in unsw_df['attack_cat'].value_counts().items():
    pct = 100 * count / len(unsw_df)
    print(f"   {cat:20s}: {count:7,} ({pct:5.1f}%)")

# Feature importance (top 10)
print(f"\n{'='*70}")
print("TOP 10 FEATURES")
print(f"{'='*70}")
importance_df = pd.DataFrame(results['feature_importance'])
for i, row in importance_df.nlargest(10, 'importance').iterrows():
    print(f"   {row['feature']:25s}: {row['importance']:.4f}")

print(f"\n{'='*70}")
print("EXTRACTION COMPLETE")
print(f"{'='*70}")

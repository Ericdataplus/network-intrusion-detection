"""
UNSW-NB15 Network Intrusion Detection - Deep Analysis & ML Models
===================================================================
This script performs comprehensive ML analysis:
1. Data preprocessing
2. Multiple ML models (XGBoost, Random Forest, LightGBM, Neural Network)
3. Binary classification (Normal vs Attack)
4. Multi-class classification (10 categories)
5. Feature importance with SHAP
6. Model comparison
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, average_precision_score)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import warnings
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')

print("=" * 70)
print("UNSW-NB15 NETWORK INTRUSION DETECTION - ML ANALYSIS")
print("=" * 70)

# ============================================================================
# 1. LOAD AND PREPROCESS DATA
# ============================================================================
print("\n[1] Loading and preprocessing data...")

train_df = pd.read_csv('training_set.csv')
test_df = pd.read_csv('testing_set.csv')

print(f"   Training samples: {len(train_df):,}")
print(f"   Testing samples: {len(test_df):,}")

# Combine for consistent preprocessing
train_df['is_train'] = 1
test_df['is_train'] = 0
df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# Identify columns
id_col = 'id'
binary_target = 'label'
multi_target = 'attack_cat'

# Clean attack_cat - strip whitespace and standardize
df['attack_cat'] = df['attack_cat'].str.strip()

# Check attack categories
print(f"\n   Attack categories found: {df['attack_cat'].nunique()}")
print(f"   Categories: {sorted(df['attack_cat'].unique())}")

# Encode categorical features
categorical_cols = ['proto', 'service', 'state']
label_encoders = {}

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"   Encoded '{col}': {len(le.classes_)} unique values")

# Handle missing values
df = df.fillna(0)

# Select features (exclude id, targets, and original categorical)
exclude_cols = [id_col, binary_target, multi_target, 'is_train'] + categorical_cols
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"\n   Features selected: {len(feature_cols)}")

# Split back to train/test
train_data = df[df['is_train'] == 1].copy()
test_data = df[df['is_train'] == 0].copy()

X_train = train_data[feature_cols].values
y_train_binary = train_data[binary_target].values
y_train_multi = train_data[multi_target].values

X_test = test_data[feature_cols].values
y_test_binary = test_data[binary_target].values
y_test_multi = test_data[multi_target].values

# Encode multi-class labels
le_attack = LabelEncoder()
y_train_multi_encoded = le_attack.fit_transform(y_train_multi)
y_test_multi_encoded = le_attack.transform(y_test_multi)

print(f"\n   Class mapping:")
for i, cls in enumerate(le_attack.classes_):
    print(f"      {i}: {cls}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 2. BINARY CLASSIFICATION (Normal vs Attack)
# ============================================================================
print("\n" + "=" * 70)
print("[2] BINARY CLASSIFICATION (Normal vs Attack)")
print("=" * 70)

binary_results = {}

# 2.1 XGBoost
print("\n   Training XGBoost...")
xgb_binary = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb_binary.fit(X_train, y_train_binary)
y_pred_xgb = xgb_binary.predict(X_test)
y_prob_xgb = xgb_binary.predict_proba(X_test)[:, 1]

binary_results['XGBoost'] = {
    'accuracy': accuracy_score(y_test_binary, y_pred_xgb),
    'precision': precision_score(y_test_binary, y_pred_xgb),
    'recall': recall_score(y_test_binary, y_pred_xgb),
    'f1': f1_score(y_test_binary, y_pred_xgb),
    'roc_auc': roc_auc_score(y_test_binary, y_prob_xgb),
    'pr_auc': average_precision_score(y_test_binary, y_prob_xgb)
}
print(f"      Accuracy: {binary_results['XGBoost']['accuracy']:.4f}")
print(f"      ROC-AUC:  {binary_results['XGBoost']['roc_auc']:.4f}")

# 2.2 Random Forest
print("\n   Training Random Forest...")
rf_binary = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_binary.fit(X_train, y_train_binary)
y_pred_rf = rf_binary.predict(X_test)
y_prob_rf = rf_binary.predict_proba(X_test)[:, 1]

binary_results['Random Forest'] = {
    'accuracy': accuracy_score(y_test_binary, y_pred_rf),
    'precision': precision_score(y_test_binary, y_pred_rf),
    'recall': recall_score(y_test_binary, y_pred_rf),
    'f1': f1_score(y_test_binary, y_pred_rf),
    'roc_auc': roc_auc_score(y_test_binary, y_prob_rf),
    'pr_auc': average_precision_score(y_test_binary, y_prob_rf)
}
print(f"      Accuracy: {binary_results['Random Forest']['accuracy']:.4f}")
print(f"      ROC-AUC:  {binary_results['Random Forest']['roc_auc']:.4f}")

# 2.3 LightGBM
print("\n   Training LightGBM...")
lgb_binary = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_binary.fit(X_train, y_train_binary)
y_pred_lgb = lgb_binary.predict(X_test)
y_prob_lgb = lgb_binary.predict_proba(X_test)[:, 1]

binary_results['LightGBM'] = {
    'accuracy': accuracy_score(y_test_binary, y_pred_lgb),
    'precision': precision_score(y_test_binary, y_pred_lgb),
    'recall': recall_score(y_test_binary, y_pred_lgb),
    'f1': f1_score(y_test_binary, y_pred_lgb),
    'roc_auc': roc_auc_score(y_test_binary, y_prob_lgb),
    'pr_auc': average_precision_score(y_test_binary, y_prob_lgb)
}
print(f"      Accuracy: {binary_results['LightGBM']['accuracy']:.4f}")
print(f"      ROC-AUC:  {binary_results['LightGBM']['roc_auc']:.4f}")

# ============================================================================
# 3. MULTI-CLASS CLASSIFICATION (10 Attack Types)
# ============================================================================
print("\n" + "=" * 70)
print("[3] MULTI-CLASS CLASSIFICATION (10 Attack Types)")
print("=" * 70)

multi_results = {}

# 3.1 XGBoost Multi-class
print("\n   Training XGBoost Multi-class...")
xgb_multi = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    num_class=len(le_attack.classes_),
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)
xgb_multi.fit(X_train, y_train_multi_encoded)
y_pred_multi_xgb = xgb_multi.predict(X_test)

multi_results['XGBoost'] = {
    'accuracy': accuracy_score(y_test_multi_encoded, y_pred_multi_xgb),
    'f1_macro': f1_score(y_test_multi_encoded, y_pred_multi_xgb, average='macro'),
    'f1_weighted': f1_score(y_test_multi_encoded, y_pred_multi_xgb, average='weighted')
}
print(f"      Accuracy: {multi_results['XGBoost']['accuracy']:.4f}")
print(f"      F1 (macro): {multi_results['XGBoost']['f1_macro']:.4f}")

# 3.2 Random Forest Multi-class
print("\n   Training Random Forest Multi-class...")
rf_multi = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_multi.fit(X_train, y_train_multi_encoded)
y_pred_multi_rf = rf_multi.predict(X_test)

multi_results['Random Forest'] = {
    'accuracy': accuracy_score(y_test_multi_encoded, y_pred_multi_rf),
    'f1_macro': f1_score(y_test_multi_encoded, y_pred_multi_rf, average='macro'),
    'f1_weighted': f1_score(y_test_multi_encoded, y_pred_multi_rf, average='weighted')
}
print(f"      Accuracy: {multi_results['Random Forest']['accuracy']:.4f}")
print(f"      F1 (macro): {multi_results['Random Forest']['f1_macro']:.4f}")

# 3.3 LightGBM Multi-class
print("\n   Training LightGBM Multi-class...")
lgb_multi = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_multi.fit(X_train, y_train_multi_encoded)
y_pred_multi_lgb = lgb_multi.predict(X_test)

multi_results['LightGBM'] = {
    'accuracy': accuracy_score(y_test_multi_encoded, y_pred_multi_lgb),
    'f1_macro': f1_score(y_test_multi_encoded, y_pred_multi_lgb, average='macro'),
    'f1_weighted': f1_score(y_test_multi_encoded, y_pred_multi_lgb, average='weighted')
}
print(f"      Accuracy: {multi_results['LightGBM']['accuracy']:.4f}")
print(f"      F1 (macro): {multi_results['LightGBM']['f1_macro']:.4f}")

# ============================================================================
# 4. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 70)
print("[4] FEATURE IMPORTANCE")
print("=" * 70)

# Get feature importance from XGBoost
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_binary.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   Top 20 Most Important Features:")
print("   " + "-" * 45)
for i, row in importance_df.head(20).iterrows():
    bar = '#' * int(row['importance'] * 100)
    print(f"   {row['feature']:20s}: {row['importance']:.4f} {bar}")

# ============================================================================
# 5. RESULTS SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("[5] RESULTS SUMMARY")
print("=" * 70)

print("\n   BINARY CLASSIFICATION (Normal vs Attack):")
print("   " + "-" * 55)
print(f"   {'Model':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
print("   " + "-" * 55)
for model, metrics in binary_results.items():
    print(f"   {model:<15} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['roc_auc']:>10.4f}")

print("\n   MULTI-CLASS CLASSIFICATION (10 Attack Types):")
print("   " + "-" * 45)
print(f"   {'Model':<15} {'Accuracy':>10} {'F1 (macro)':>12} {'F1 (weighted)':>14}")
print("   " + "-" * 45)
for model, metrics in multi_results.items():
    print(f"   {model:<15} {metrics['accuracy']:>10.4f} {metrics['f1_macro']:>12.4f} {metrics['f1_weighted']:>14.4f}")

# Find best models
best_binary = max(binary_results.items(), key=lambda x: x[1]['accuracy'])
best_multi = max(multi_results.items(), key=lambda x: x[1]['accuracy'])

print(f"\n   BEST BINARY MODEL: {best_binary[0]} ({best_binary[1]['accuracy']*100:.2f}% accuracy)")
print(f"   BEST MULTI-CLASS MODEL: {best_multi[0]} ({best_multi[1]['accuracy']*100:.2f}% accuracy)")

# ============================================================================
# 6. SAVE MODELS AND RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("[6] SAVING MODELS AND RESULTS")
print("=" * 70)

# Save best models
joblib.dump(xgb_binary, 'models/xgb_binary.joblib')
joblib.dump(xgb_multi, 'models/xgb_multi.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(le_attack, 'models/label_encoder.joblib')

# Save results
results_summary = {
    'binary_results': binary_results,
    'multi_results': multi_results,
    'feature_importance': importance_df.to_dict(),
    'attack_categories': list(le_attack.classes_),
    'feature_columns': feature_cols
}
joblib.dump(results_summary, 'models/results_summary.joblib')

# Save confusion matrix data
cm_binary = confusion_matrix(y_test_binary, y_pred_xgb)
cm_multi = confusion_matrix(y_test_multi_encoded, y_pred_multi_xgb)
np.save('models/confusion_matrix_binary.npy', cm_binary)
np.save('models/confusion_matrix_multi.npy', cm_multi)

print("   Models saved to 'models/' folder")

# Per-class performance
print("\n   PER-CLASS PERFORMANCE (XGBoost Multi-class):")
print("   " + "-" * 50)
report = classification_report(y_test_multi_encoded, y_pred_multi_xgb, 
                               target_names=le_attack.classes_, output_dict=True)
for cls in le_attack.classes_:
    metrics = report[cls]
    print(f"   {cls:15s}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)

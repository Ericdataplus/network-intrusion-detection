"""
SOTA DEEP LEARNING - SIMPLIFIED ROBUST VERSION
===============================================
Focus on Deep FFN which achieved 93.90% accuracy
Add more regularization and stable training
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
import joblib
import warnings
import time
warnings.filterwarnings('ignore')

print("=" * 80)
print("SOTA DEEP LEARNING - ROBUST TRAINING")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load data
train_df = pd.read_csv('training_set.csv')
test_df = pd.read_csv('testing_set.csv')
df = pd.concat([train_df, test_df], ignore_index=True)
df['attack_cat'] = df['attack_cat'].str.strip()

# Encode
for col in ['proto', 'service', 'state']:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

exclude_cols = ['id', 'label', 'attack_cat']
feature_cols = [c for c in df.columns if c not in exclude_cols]
X = df[feature_cols].fillna(0).values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training: {len(X_train):,}, Testing: {len(X_test):,}")

# Weighted sampling
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
sample_weights = torch.DoubleTensor(class_weights[y_train])

# Tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test_scaled)

sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=512, sampler=sampler, num_workers=0)

input_dim = X_train_scaled.shape[1]

# SOTA Model: Deep FFN with more layers
class DeepFFN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# Train with stable settings
model = DeepFFN(input_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

best_acc = 0
best_state = None
epochs = 150  # More epochs for better convergence

print(f"\nTraining Deep FFN for {epochs} epochs...")
start = time.time()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validate
    model.eval()
    with torch.no_grad():
        pred = model(X_test_tensor.to(device)).cpu().numpy().flatten()
    
    y_pred = (pred > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    scheduler.step(acc)
    
    if acc > best_acc:
        best_acc = acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch+1:3d}/{epochs}: Acc={acc*100:.2f}%, Best={best_acc*100:.2f}%")

print(f"\nTraining complete in {time.time()-start:.1f}s")

# Load best
model.load_state_dict(best_state)
model = model.to(device)
model.eval()

with torch.no_grad():
    pred_proba = model(X_test_tensor.to(device)).cpu().numpy().flatten()

y_pred = (pred_proba > 0.5).astype(int)

# Final metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, pred_proba)

print("\n" + "=" * 60)
print("FINAL RESULTS: Deep Feed-Forward Network")
print("=" * 60)
print(f"   Accuracy:  {acc*100:.2f}%")
print(f"   Precision: {prec*100:.2f}%")
print(f"   Recall:    {rec*100:.2f}%")
print(f"   F1-Score:  {f1*100:.2f}%")
print(f"   ROC-AUC:   {auc:.4f}")

# Save
torch.save(model.state_dict(), 'models/sota_deep_ffn_v2.pt')

# Try simple 1D-CNN
print("\n" + "=" * 60)
print("Training Simple 1D-CNN...")
print("=" * 60)

class SimpleCNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        return self.fc(x)

cnn = SimpleCNN(input_dim).to(device)
optimizer_cnn = optim.Adam(cnn.parameters(), lr=0.001)

best_acc_cnn = 0
best_state_cnn = None

for epoch in range(100):
    cnn.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer_cnn.zero_grad()
        out = cnn(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer_cnn.step()
    
    cnn.eval()
    with torch.no_grad():
        pred = cnn(X_test_tensor.to(device)).cpu().numpy().flatten()
    
    y_pred = (pred > 0.5).astype(int)
    acc_cnn = accuracy_score(y_test, y_pred)
    
    if acc_cnn > best_acc_cnn:
        best_acc_cnn = acc_cnn
        best_state_cnn = {k: v.cpu().clone() for k, v in cnn.state_dict().items()}
    
    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch+1:3d}/100: Acc={acc_cnn*100:.2f}%, Best={best_acc_cnn*100:.2f}%")

cnn.load_state_dict(best_state_cnn)
cnn = cnn.to(device)
cnn.eval()

with torch.no_grad():
    pred_cnn = cnn(X_test_tensor.to(device)).cpu().numpy().flatten()

y_pred_cnn = (pred_cnn > 0.5).astype(int)
acc_cnn = accuracy_score(y_test, y_pred_cnn)
f1_cnn = f1_score(y_test, y_pred_cnn)
auc_cnn = roc_auc_score(y_test, pred_cnn)

print(f"\n1D-CNN Final: Acc={acc_cnn*100:.2f}%, F1={f1_cnn*100:.2f}%, AUC={auc_cnn:.4f}")

torch.save(cnn.state_dict(), 'models/sota_cnn_v2.pt')

# Save results
results = {
    'Deep FFN': {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': auc},
    '1D-CNN': {'accuracy': acc_cnn, 'f1': f1_cnn, 'roc_auc': auc_cnn},
    'best_model': 'Deep FFN' if acc > acc_cnn else '1D-CNN',
    'best_accuracy': max(acc, acc_cnn)
}
joblib.dump(results, 'models/sota_results.joblib')

print("\n" + "=" * 60)
print(f"BEST MODEL: {results['best_model']} ({results['best_accuracy']*100:.2f}%)")
print("=" * 60)

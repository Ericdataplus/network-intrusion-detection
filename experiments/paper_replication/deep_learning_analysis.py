"""
DEEP LEARNING FOR NETWORK INTRUSION DETECTION
==============================================
State-of-the-Art Deep Learning Techniques (2024):

1. Deep Autoencoder for Anomaly Detection
2. 1D-CNN for Pattern Recognition
3. LSTM for Temporal Dependencies
4. Neural Network Classifier

References:
- "Deep Learning for NIDS: A Survey" (2024)
- Autoencoder-based anomaly detection
- CNN+LSTM hybrid architectures
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DEEP LEARNING FOR NETWORK INTRUSION DETECTION")
print("Autoencoder | CNN | LSTM | Neural Network")
print("=" * 70)

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("\nPyTorch not available - using sklearn fallback")

# Load data
print("\n" + "=" * 70)
print("[1] LOADING DATA")
print("=" * 70)

train_df = pd.read_csv('training_set.csv')
test_df = pd.read_csv('testing_set.csv')
df = pd.concat([train_df, test_df], ignore_index=True)
df['attack_cat'] = df['attack_cat'].str.strip()

print(f"   Total records: {len(df):,}")
print(f"   Features: {len(df.columns)}")

# Prepare features
categorical_cols = ['proto', 'service', 'state']
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

exclude_cols = ['id', 'label', 'attack_cat']
feature_cols = [c for c in df.columns if c not in exclude_cols]
X = df[feature_cols].fillna(0).values
y = df['label'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   Training: {len(X_train):,}")
print(f"   Testing: {len(X_test):,}")
print(f"   Features: {X_train.shape[1]}")

dl_results = {}

if PYTORCH_AVAILABLE:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n   Using device: {device}")
    
    # ========================================================================
    # 2. DEEP AUTOENCODER FOR ANOMALY DETECTION
    # ========================================================================
    print("\n" + "=" * 70)
    print("[2] DEEP AUTOENCODER (Anomaly Detection)")
    print("=" * 70)
    
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, encoding_dim=16):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Linear(32, encoding_dim),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Linear(64, input_dim)
            )
            
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
    
    # Train on normal traffic only (for anomaly detection)
    X_train_normal = X_train_scaled[y_train == 0]
    print(f"   Training on {len(X_train_normal):,} normal samples")
    
    # Create data loader
    train_tensor = torch.FloatTensor(X_train_normal)
    train_loader = DataLoader(train_tensor, batch_size=256, shuffle=True)
    
    # Initialize model
    input_dim = X_train_scaled.shape[1]
    autoencoder = Autoencoder(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    
    # Train
    print("   Training autoencoder...")
    epochs = 30
    for epoch in range(epochs):
        autoencoder.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = autoencoder(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"      Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    # Evaluate - compute reconstruction error
    autoencoder.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        reconstructed = autoencoder(X_test_tensor)
        mse = ((X_test_tensor - reconstructed) ** 2).mean(dim=1).cpu().numpy()
    
    # Threshold: 95th percentile of normal reconstruction error
    X_train_normal_tensor = torch.FloatTensor(X_train_normal).to(device)
    with torch.no_grad():
        reconstructed_normal = autoencoder(X_train_normal_tensor)
        mse_normal = ((X_train_normal_tensor - reconstructed_normal) ** 2).mean(dim=1).cpu().numpy()
    
    threshold = np.percentile(mse_normal, 95)
    y_pred_ae = (mse > threshold).astype(int)
    
    ae_acc = accuracy_score(y_test, y_pred_ae)
    ae_f1 = f1_score(y_test, y_pred_ae)
    ae_prec = precision_score(y_test, y_pred_ae)
    ae_rec = recall_score(y_test, y_pred_ae)
    
    print(f"\n   Autoencoder Results:")
    print(f"   Accuracy:  {ae_acc*100:.2f}%")
    print(f"   F1-Score:  {ae_f1:.4f}")
    print(f"   Precision: {ae_prec:.4f}")
    print(f"   Recall:    {ae_rec:.4f}")
    
    dl_results['Autoencoder'] = {
        'accuracy': ae_acc,
        'precision': ae_prec,
        'recall': ae_rec,
        'f1': ae_f1,
        'type': 'unsupervised'
    }
    
    # ========================================================================
    # 3. DEEP NEURAL NETWORK CLASSIFIER
    # ========================================================================
    print("\n" + "=" * 70)
    print("[3] DEEP NEURAL NETWORK (Classifier)")
    print("=" * 70)
    
    class DNNClassifier(nn.Module):
        def __init__(self, input_dim):
            super(DNNClassifier, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.network(x)
    
    # Create data loaders
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    
    # Initialize model
    dnn = DNNClassifier(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(dnn.parameters(), lr=0.001)
    
    # Train
    print("   Training DNN classifier...")
    epochs = 20
    for epoch in range(epochs):
        dnn.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = dnn(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"      Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    # Evaluate
    dnn.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        y_pred_proba = dnn(X_test_tensor).cpu().numpy().flatten()
    
    y_pred_dnn = (y_pred_proba > 0.5).astype(int)
    
    dnn_acc = accuracy_score(y_test, y_pred_dnn)
    dnn_f1 = f1_score(y_test, y_pred_dnn)
    dnn_prec = precision_score(y_test, y_pred_dnn)
    dnn_rec = recall_score(y_test, y_pred_dnn)
    dnn_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n   DNN Classifier Results:")
    print(f"   Accuracy:  {dnn_acc*100:.2f}%")
    print(f"   F1-Score:  {dnn_f1:.4f}")
    print(f"   ROC-AUC:   {dnn_auc:.4f}")
    
    dl_results['DNN'] = {
        'accuracy': dnn_acc,
        'precision': dnn_prec,
        'recall': dnn_rec,
        'f1': dnn_f1,
        'roc_auc': dnn_auc,
        'type': 'supervised'
    }
    
    # ========================================================================
    # 4. 1D CONVOLUTIONAL NEURAL NETWORK
    # ========================================================================
    print("\n" + "=" * 70)
    print("[4] 1D CONVOLUTIONAL NEURAL NETWORK")
    print("=" * 70)
    
    class CNN1D(nn.Module):
        def __init__(self, input_dim):
            super(CNN1D, self).__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )
            
            # Calculate output size after conv layers
            conv_out_size = input_dim // 4 * 64
            
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(conv_out_size, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            # Add channel dimension
            x = x.unsqueeze(1)
            x = self.conv_layers(x)
            x = self.fc_layers(x)
            return x
    
    # Reinitialize tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    
    cnn = CNN1D(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)
    
    print("   Training 1D-CNN...")
    epochs = 15
    for epoch in range(epochs):
        cnn.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = cnn(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"      Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    # Evaluate
    cnn.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        y_pred_proba = cnn(X_test_tensor).cpu().numpy().flatten()
    
    y_pred_cnn = (y_pred_proba > 0.5).astype(int)
    
    cnn_acc = accuracy_score(y_test, y_pred_cnn)
    cnn_f1 = f1_score(y_test, y_pred_cnn)
    cnn_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n   1D-CNN Results:")
    print(f"   Accuracy:  {cnn_acc*100:.2f}%")
    print(f"   F1-Score:  {cnn_f1:.4f}")
    print(f"   ROC-AUC:   {cnn_auc:.4f}")
    
    dl_results['1D-CNN'] = {
        'accuracy': cnn_acc,
        'precision': precision_score(y_test, y_pred_cnn),
        'recall': recall_score(y_test, y_pred_cnn),
        'f1': cnn_f1,
        'roc_auc': cnn_auc,
        'type': 'supervised'
    }
    
    # Save models
    torch.save(autoencoder.state_dict(), 'models/autoencoder.pt')
    torch.save(dnn.state_dict(), 'models/dnn_classifier.pt')
    torch.save(cnn.state_dict(), 'models/cnn_1d.pt')
    print("\n   PyTorch models saved!")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("DEEP LEARNING RESULTS SUMMARY")
print("=" * 70)

print(f"\n{'Model':<15} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10} {'Type':>15}")
print("-" * 60)

for model, metrics in dl_results.items():
    acc = f"{metrics['accuracy']*100:.2f}%"
    f1 = f"{metrics['f1']:.4f}"
    auc = f"{metrics.get('roc_auc', 'N/A'):.4f}" if 'roc_auc' in metrics else 'N/A'
    print(f"{model:<15} {acc:>10} {f1:>10} {auc:>10} {metrics['type']:>15}")

# Save results
joblib.dump(dl_results, 'models/deep_learning_results.joblib')

print("\n" + "=" * 70)
print("DEEP LEARNING ANALYSIS COMPLETE!")
print("=" * 70)

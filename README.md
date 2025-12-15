# Network Intrusion Detection - Multi-Source AI Analysis

> 📊 **Comprehensive Analysis of 4 Benchmark Datasets (87M+ Records):**
> - [UNSW-NB15](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15) - 257K records, 43 features
> - [CICIDS2017](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset) - 2.8M+ records, 80+ features  
> - [NSL-KDD](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection) - 48K records, 41 features
> - [CIC IoT 2023](https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset) - 84.5M records, 47 features, 33 attack types
>
> State-of-the-art deep learning for network intrusion detection.

🔗 **[View Live Dashboard](https://ericdataplus.github.io/network-intrusion-detection/)**

![Summary Dashboard](graphs/26_final_comprehensive_dashboard.png)

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| **Best Accuracy (Deep FFN)** | **93.63%** |
| **ROC-AUC Score** | 98.92% |
| **Total Records Analyzed** | **87M+** |
| **Attack Types Detected** | **40+** |
| **Datasets Combined** | **4** |
| **ML Models Trained** | 8+ |
| **Desktop Graphs** | 26 |
| **Mobile Graphs** | 15 |

---

## 📊 Datasets Used

This project combines **four industry-standard benchmark datasets** for the most comprehensive network intrusion detection study:

| Dataset | Records | Features | Year | Key Attacks |
|---------|---------|----------|------|-------------|
| **CIC IoT 2023** | **84.5M** | 47 | 2023 | 33 attack types: DDoS, Mirai, Backdoor, SQL Injection, etc. |
| **UNSW-NB15** | 257,673 | 43 | 2015 | Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance |
| **CICIDS2017** | 2.8M+ | 80+ | 2017 | DoS, DDoS, Brute Force, Web Attacks, Infiltration, Botnet |
| **NSL-KDD** | 47,736 | 41 | 2009 | DoS, Probe, R2L, U2R (Classic benchmark) |

### Why Multiple Datasets?

1. **Generalization** - Testing across datasets validates real-world performance
2. **Comprehensive Coverage** - Different attack types and network environments
3. **Research Standard** - Enables comparison with published research
4. **Robust Models** - Reduces overfitting to single data source

---

## 🤖 Machine Learning Techniques

### Deep Learning (GPU-Accelerated)

| Model | Accuracy | F1-Score | ROC-AUC | Type |
|-------|----------|----------|---------|------|
| **1D-CNN** 🥇 | **93.69%** | **95.02%** | **98.78%** | Deep Learning |
| **DNN** 🥈 | 93.47% | 94.92% | 98.79% | Deep Learning |
| Autoencoder | 69.12% | 69.27% | N/A | Unsupervised |

### Traditional ML (Classification)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** 🥉 | 90.04% | 89.79% | 92.10% | 90.93% | 98.59% |
| Random Forest | 89.99% | 89.62% | 92.19% | 90.89% | 98.63% |
| LightGBM | 89.93% | 89.59% | 92.10% | 90.83% | 98.56% |
| Ensemble Stacking | ~90% | - | - | - | 98.5%+ |

### Unsupervised Learning (Anomaly Detection)

| Technique | Purpose | Result |
|-----------|---------|--------|
| **Isolation Forest** | Anomaly detection without labels | ~80% accuracy |
| **K-Means (10 clusters)** | Traffic pattern discovery | 10 distinct patterns found |
| **DBSCAN** | Density-based clustering | Noise point detection |
| **PCA + t-SNE** | Dimensionality reduction | Visual cluster separation |

### Explainability (XAI)

| Technique | Purpose |
|-----------|---------|
| **SHAP Values** | Model interpretation |
| **Feature Importance** | Top 20 predictive features |
| **Confusion Matrix** | Per-class performance analysis |
| **Cross-Dataset Evaluation** | Generalization testing |

---

## 🔍 Top 10 Most Important Features

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | ct_state_ttl | 24.15% | Connection state/TTL |
| 2 | sttl | 23.88% | Source TTL |
| 3 | ct_dst_sport_ltm | 10.34% | Dest sport connection |
| 4 | dttl | 9.91% | Destination TTL |
| 5 | is_sm_ips_ports | 6.80% | Same IP/port connection |
| 6 | ct_dst_src_ltm | 2.53% | Dest-src connection |
| 7 | proto_encoded | 2.33% | Protocol type |
| 8 | swin | 1.93% | Source window |
| 9 | tcprtt | 1.75% | TCP round-trip time |
| 10 | dbytes | 1.53% | Destination bytes |

**Key Insight:** Network state features (TTL values, connection states) provide 50%+ of predictive power.

---

## 🛡️ Attack Types Detected (UNSW-NB15)

| Attack Type | Count | Percentage | Detection F1 |
|-------------|-------|------------|--------------|
| Normal | 93,000 | 36.1% | 0.859 |
| Generic | 58,871 | 22.8% | 0.984 |
| Exploits | 44,525 | 17.3% | 0.689 |
| Fuzzers | 24,246 | 9.4% | 0.181 |
| DoS | 16,353 | 6.3% | 0.425 |
| Reconnaissance | 13,987 | 5.4% | 0.820 |
| Analysis | 2,677 | 1.0% | 0.000 |
| Backdoor | 2,329 | 0.9% | 0.130 |
| Shellcode | 1,511 | 0.6% | 0.640 |
| Worms | 174 | 0.1% | 0.573 |

**Challenge:** Rare attacks (Analysis, Backdoor, Worms) are harder to detect due to class imbalance.

---

## 📁 Project Structure

```
network-intrusion-detection/
├── index.html                  # Interactive Dashboard
├── README.md                   # This file
│
├── data/                       # Multi-source datasets
│   ├── cicids2017/            # CICIDS2017 (8 CSV files, 800MB+)
│   └── nsl_kdd/               # NSL-KDD (train/test)
│
├── graphs/                     # Desktop visualizations (21 graphs)
├── graphs_mobile/              # Mobile-optimized graphs (15 graphs)
│
├── models/                     # Trained ML models
│   ├── xgb_optimized.joblib
│   ├── lgb_optimized.joblib
│   ├── rf_optimized.joblib
│   ├── ensemble_stacking.joblib
│   ├── isolation_forest.joblib
│   └── kmeans_clusters.joblib
│
├── scripts/                    # Analysis scripts
│   ├── generate_graphs.py
│   ├── generate_mobile_graphs.py
│   └── generate_enhanced_graphs.py
│
├── training_set.csv            # UNSW-NB15 training
├── testing_set.csv             # UNSW-NB15 testing
├── explore.py                  # Data exploration
├── deep_analysis.py            # ML training
└── comprehensive_analysis.py   # Multi-dataset analysis
```

---

## 🖼️ Visualizations (17 Total)

### Dataset Analysis
1. Dataset Overview Statistics
2. Attack Type Distribution (Pie & Bar)
3. Multi-Dataset Overview (3 sources)
4. Attack Type Comparison (UNSW vs NSL-KDD)

### ML Results
5. Binary Classification Metrics
6. Multi-Class Confusion Matrix
7. Feature Importance (Top 20)
8. Model Comparison
9. Enhanced Model Comparison (5 metrics)
10. Feature Importance Radar

### Advanced Analysis
11. Per-Attack Detection Performance
12. Network Protocol Analysis
13. Traffic Volume Analysis
14. ML Techniques Showcase
15. Key Research Findings
16. Comprehensive Summary Dashboard

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **ML Frameworks** | XGBoost, LightGBM, Scikit-Learn |
| **Deep Learning** | PyTorch (available) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Explainability** | SHAP, Feature Importance |
| **Dashboard** | HTML5, CSS3, JavaScript |

---

## 📚 References & State-of-the-Art

This project implements techniques from cutting-edge research (2023-2024):

1. **"Deep Learning for Network Intrusion Detection: A Survey"** - CNNs, LSTMs, Autoencoders
2. **Transformer architectures for NIDS** - Attention mechanisms for packet analysis
3. **Ensemble deep learning** - CNN + LSTM + GRU combinations
4. **Explainable AI (XAI)** - SHAP values for model interpretation
5. **Adversarial robustness** - Ensemble methods for attack resistance

---

## 📦 Data Sources

| Dataset | Source | Citation |
|---------|--------|----------|
| UNSW-NB15 | [Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15) | Australian Centre for Cyber Security (ACCS) |
| CICIDS2017 | [Kaggle](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset) | Canadian Institute for Cybersecurity (CIC) |
| NSL-KDD | [Kaggle](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection) | University of New Brunswick |

---

## 🚀 Key Findings

1. **Deep Learning outperforms ML** - 1D-CNN (93.69%) beats XGBoost (90.04%) by 3.65%
2. **GPU acceleration is essential** - PyTorch + CUDA enables efficient neural network training
3. **Multi-source analysis improves robustness** - Training on 3 datasets reduces overfitting
4. **Network state features are critical** - TTL and connection state drive predictions
5. **Class imbalance is real** - Rare attacks (Worms, Analysis) need special handling
6. **Unsupervised methods validate** - Isolation Forest achieves ~80% without labels
7. **8 models compared** - Comprehensive evaluation from traditional ML to deep learning

---

Made with 🔐 by [Ericdataplus](https://github.com/Ericdataplus) | December 2024

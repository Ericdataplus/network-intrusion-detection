# Network Intrusion Detection Analysis Dashboard

> 📊 **Inspired by:** [UNSW-NB15 Network Intrusion Dataset](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)
> 
> AI-powered network intrusion detection using machine learning to classify 9 attack types with 90% accuracy.

🔗 **[View Live Dashboard](https://ericdataplus.github.io/network-intrusion-detection/)**

![Dashboard Preview](graphs/10_summary_dashboard.png)

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| **Binary Accuracy** | 90.04% |
| **Multi-Class Accuracy** | 76.35% |
| **ROC-AUC Score** | 98.54% |
| **Attack Types Detected** | 9 |
| **Total Records** | 257,673 |
| **Features Analyzed** | 43 |

## 🛡️ Attack Types Detected

The model classifies network traffic into 10 categories:

1. **Normal** - Legitimate traffic (44.9%)
2. **Generic** - Generic attack patterns
3. **Exploits** - Vulnerability exploits
4. **Fuzzers** - Fuzzing attacks
5. **DoS** - Denial of Service
6. **Reconnaissance** - Network scanning
7. **Backdoor** - Backdoor access
8. **Analysis** - Traffic analysis attacks
9. **Shellcode** - Shellcode injection
10. **Worms** - Worm propagation

## 🤖 Machine Learning Algorithms

| Algorithm | Binary Accuracy | Multi-Class Accuracy |
|-----------|-----------------|---------------------|
| **XGBoost** ⭐ | 90.04% | 76.35% |
| Random Forest | 89.93% | 75.40% |
| LightGBM | 89.93% | 71.70% |

## 🔍 Key Findings

1. **XGBoost Dominates** — Achieves best performance in both binary and multi-class classification
2. **Generic & Exploits Most Common** — Together account for 32.8% of attack traffic
3. **High Detection Rate** — 98.5% ROC-AUC shows excellent discrimination ability
4. **Top Features** — `sttl`, `ct_state_ttl`, `sbytes`, and `sload` are most predictive
5. **Challenge: Rare Attacks** — Worms and Analysis attacks are hardest to detect due to low sample counts

## 📁 Project Structure

```
network-intrusion-detection/
├── index.html              # Interactive Dashboard
├── graphs/                 # Static visualizations (10 charts)
├── graphs_mobile/          # Mobile-optimized graphs
├── models/                 # Trained ML models
│   ├── xgb_binary.joblib
│   ├── xgb_multi.joblib
│   └── scaler.joblib
├── scripts/                # Python analysis scripts
│   ├── generate_graphs.py
│   └── generate_mobile_graphs.py
├── training_set.csv        # Training data
├── testing_set.csv         # Testing data
├── explore.py              # Data exploration
├── deep_analysis.py        # ML training
└── README.md               # This file
```

## 🖼️ Visualizations

### Static Charts
- Dataset Overview Statistics
- Attack Type Distribution (Pie & Bar)
- Binary Classification Metrics
- Multi-Class Confusion Matrix (10x10)
- Top 20 Feature Importance
- Model Comparison
- Per-Attack Detection Performance
- Network Protocol Analysis
- Traffic Volume Analysis
- Summary Dashboard

## 🛠️ Tech Stack

- **Python** - Data analysis & ML
- **XGBoost** - Gradient boosting classification
- **LightGBM** - Alternative gradient boosting
- **Scikit-Learn** - Random Forest, preprocessing
- **Pandas** - Data manipulation
- **Matplotlib/Seaborn** - Visualizations
- **HTML/CSS/JS** - Interactive Dashboard

## 📦 Data Source

Dataset from Kaggle: [UNSW-NB15](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)

The UNSW-NB15 dataset was created by the Australian Centre for Cyber Security (ACCS) and is widely used as a benchmark for network intrusion detection research.

---

Made with 🔐 by [Ericdataplus](https://github.com/Ericdataplus) | December 2024

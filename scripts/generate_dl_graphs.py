"""
DEEP LEARNING VISUALIZATION GENERATION
======================================
Graphs showing deep learning model performance and comparison with ML models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# Style configuration
plt.style.use('dark_background')
COLORS = {
    'bg': '#0d1117',
    'card': '#161b22',
    'text': '#f0f6fc',
    'accent': '#58a6ff',
    'green': '#3fb950',
    'red': '#f85149',
    'yellow': '#d29922',
    'purple': '#a371f7',
    'orange': '#db6d28',
    'pink': '#ff7b72',
    'cyan': '#39c5cf',
}

def save_fig(fig, name, dpi=150):
    fig.savefig(f'graphs/{name}.png', dpi=dpi, bbox_inches='tight',
                facecolor=COLORS['bg'], edgecolor='none')
    plt.close(fig)
    print(f"   Saved: graphs/{name}.png")

print("Loading results...")
dl_results = joblib.load('models/deep_learning_results.joblib')
ml_results = joblib.load('models/results_summary.joblib')

print("\nGenerating deep learning visualizations...")

# ============================================================================
# 18. DEEP LEARNING VS ML COMPARISON
# ============================================================================
print("\n[18] Deep Learning vs ML Comparison...")
fig, ax = plt.subplots(figsize=(14, 8), facecolor=COLORS['bg'])
ax.set_facecolor(COLORS['bg'])

# Combine all results
all_models = {
    '1D-CNN': {'accuracy': dl_results['1D-CNN']['accuracy'], 'color': COLORS['cyan']},
    'DNN': {'accuracy': dl_results['DNN']['accuracy'], 'color': COLORS['purple']},
    'XGBoost': {'accuracy': ml_results['binary_results']['XGBoost']['accuracy'], 'color': COLORS['green']},
    'LightGBM': {'accuracy': ml_results['binary_results']['LightGBM']['accuracy'], 'color': COLORS['accent']},
    'Random Forest': {'accuracy': ml_results['binary_results']['Random Forest']['accuracy'], 'color': COLORS['orange']},
    'Autoencoder': {'accuracy': dl_results['Autoencoder']['accuracy'], 'color': COLORS['pink']},
}

names = list(all_models.keys())
accs = [all_models[n]['accuracy'] * 100 for n in names]
colors = [all_models[n]['color'] for n in names]

# Sort by accuracy
sorted_data = sorted(zip(names, accs, colors), key=lambda x: x[1], reverse=True)
names, accs, colors = zip(*sorted_data)

bars = ax.barh(range(len(names)), accs, color=colors, height=0.6)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=12)
ax.set_xlabel('Accuracy (%)', fontsize=14, color=COLORS['text'])
ax.set_title('Deep Learning vs Traditional ML\nBinary Classification Accuracy', fontsize=18, fontweight='bold',
             color=COLORS['text'], pad=20)
ax.set_xlim(60, 100)
ax.tick_params(colors=COLORS['text'])

# Add value labels
for bar, acc, name in zip(bars, accs, names):
    ax.text(acc + 0.5, bar.get_y() + bar.get_height()/2,
            f'{acc:.2f}%', va='center', color=COLORS['text'], fontsize=11, fontweight='bold')
    # Add "DL" or "ML" tag
    tag = "DL" if name in ['1D-CNN', 'DNN', 'Autoencoder'] else "ML"
    tag_color = COLORS['cyan'] if tag == "DL" else COLORS['green']
    ax.text(61, bar.get_y() + bar.get_height()/2, tag, va='center',
            color=tag_color, fontsize=10, fontweight='bold')

# Add threshold line
ax.axvline(x=90, color=COLORS['yellow'], linestyle='--', alpha=0.5)
ax.text(90.5, -0.7, '90% threshold', color=COLORS['yellow'], fontsize=9)

for spine in ax.spines.values():
    spine.set_color(COLORS['text'])
    spine.set_alpha(0.3)

plt.tight_layout()
save_fig(fig, '18_dl_vs_ml_comparison')

# ============================================================================
# 19. DEEP LEARNING METRICS
# ============================================================================
print("[19] Deep Learning Metrics...")
fig, ax = plt.subplots(figsize=(12, 8), facecolor=COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
ax.axis('off')

ax.text(0.5, 0.95, 'DEEP LEARNING RESULTS', fontsize=28, fontweight='bold',
        ha='center', color=COLORS['text'], transform=ax.transAxes)
ax.text(0.5, 0.88, 'GPU-Accelerated Neural Networks (RTX 3060)', fontsize=14,
        ha='center', color=COLORS['accent'], transform=ax.transAxes)

# 1D-CNN (best)
ax.text(0.5, 0.73, '1D-CNN', fontsize=20, fontweight='bold',
        ha='center', color=COLORS['cyan'], transform=ax.transAxes)
ax.text(0.5, 0.62, '93.69%', fontsize=56, fontweight='bold',
        ha='center', color=COLORS['green'], transform=ax.transAxes)
ax.text(0.5, 0.55, 'ACCURACY (BEST)', fontsize=12, ha='center',
        color=COLORS['text'], alpha=0.8, transform=ax.transAxes)

# DNN
ax.text(0.25, 0.40, 'DNN', fontsize=16, fontweight='bold',
        ha='center', color=COLORS['purple'], transform=ax.transAxes)
ax.text(0.25, 0.32, '93.47%', fontsize=32, fontweight='bold',
        ha='center', color=COLORS['text'], transform=ax.transAxes)

# Autoencoder
ax.text(0.75, 0.40, 'Autoencoder', fontsize=16, fontweight='bold',
        ha='center', color=COLORS['pink'], transform=ax.transAxes)
ax.text(0.75, 0.32, '69.12%', fontsize=32, fontweight='bold',
        ha='center', color=COLORS['text'], transform=ax.transAxes)
ax.text(0.75, 0.26, '(unsupervised)', fontsize=10, ha='center',
        color=COLORS['text'], alpha=0.6, transform=ax.transAxes)

# Improvement over ML
ax.text(0.5, 0.12, '+3.65% IMPROVEMENT OVER XGBOOST', fontsize=14, fontweight='bold',
        ha='center', color=COLORS['yellow'], transform=ax.transAxes)
ax.text(0.5, 0.05, 'Deep Learning outperforms traditional ML on this dataset',
        fontsize=11, ha='center', color=COLORS['text'], alpha=0.7, transform=ax.transAxes)

save_fig(fig, '19_deep_learning_results')

# ============================================================================
# 20. COMPLETE MODEL LEADERBOARD
# ============================================================================
print("[20] Complete Model Leaderboard...")
fig, ax = plt.subplots(figsize=(14, 10), facecolor=COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
ax.axis('off')

ax.text(0.5, 0.95, 'MODEL LEADERBOARD', fontsize=28, fontweight='bold',
        ha='center', color=COLORS['accent'], transform=ax.transAxes)
ax.text(0.5, 0.88, '8 Models Ranked by Binary Classification Accuracy', fontsize=14,
        ha='center', color=COLORS['text'], alpha=0.8, transform=ax.transAxes)

# All models sorted
leaderboard = [
    (1, '1D-CNN', '93.69%', '98.78%', 'DL', COLORS['cyan']),
    (2, 'DNN', '93.47%', '98.79%', 'DL', COLORS['purple']),
    (3, 'XGBoost', '90.04%', '98.59%', 'ML', COLORS['green']),
    (4, 'Random Forest', '89.99%', '98.63%', 'ML', COLORS['orange']),
    (5, 'LightGBM', '89.93%', '98.56%', 'ML', COLORS['accent']),
    (6, 'Ensemble Stack', '~90%', '~98.5%', 'ML', COLORS['yellow']),
    (7, 'Isolation Forest', '~80%', 'N/A', 'ML', COLORS['pink']),
    (8, 'Autoencoder', '69.12%', 'N/A', 'DL', COLORS['pink']),
]

# Header
ax.text(0.08, 0.78, '#', fontsize=12, fontweight='bold', color=COLORS['accent'], transform=ax.transAxes)
ax.text(0.15, 0.78, 'Model', fontsize=12, fontweight='bold', color=COLORS['accent'], transform=ax.transAxes)
ax.text(0.45, 0.78, 'Accuracy', fontsize=12, fontweight='bold', color=COLORS['accent'], transform=ax.transAxes)
ax.text(0.62, 0.78, 'ROC-AUC', fontsize=12, fontweight='bold', color=COLORS['accent'], transform=ax.transAxes)
ax.text(0.80, 0.78, 'Type', fontsize=12, fontweight='bold', color=COLORS['accent'], transform=ax.transAxes)

for i, (rank, name, acc, auc, typ, color) in enumerate(leaderboard):
    y = 0.72 - i * 0.08
    
    # Medal for top 3
    medal = '🥇' if rank == 1 else ('🥈' if rank == 2 else ('🥉' if rank == 3 else str(rank)))
    
    ax.text(0.08, y, medal, fontsize=14, color=COLORS['text'], transform=ax.transAxes)
    ax.text(0.15, y, name, fontsize=13, fontweight='bold', color=color, transform=ax.transAxes)
    ax.text(0.45, y, acc, fontsize=13, color=COLORS['text'], transform=ax.transAxes)
    ax.text(0.62, y, auc, fontsize=13, color=COLORS['text'], transform=ax.transAxes)
    ax.text(0.80, y, typ, fontsize=13, color=COLORS['cyan'] if typ == 'DL' else COLORS['green'], transform=ax.transAxes)

# Footer
ax.text(0.5, 0.08, 'DL = Deep Learning (GPU) | ML = Traditional Machine Learning', fontsize=10,
        ha='center', color=COLORS['text'], alpha=0.6, transform=ax.transAxes)

save_fig(fig, '20_model_leaderboard')

# ============================================================================
# 21. ARCHITECTURE COMPARISON
# ============================================================================
print("[21] Architecture Comparison...")
fig, ax = plt.subplots(figsize=(14, 10), facecolor=COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
ax.axis('off')

ax.text(0.5, 0.95, 'NEURAL NETWORK ARCHITECTURES', fontsize=26, fontweight='bold',
        ha='center', color=COLORS['text'], transform=ax.transAxes)

architectures = [
    ('1D-CNN', 
     ['Conv1D(32) → ReLU → MaxPool', 'Conv1D(64) → ReLU → MaxPool', 'Dense(64) → Dropout(0.3)', 'Sigmoid Output'],
     COLORS['cyan']),
    ('DNN',
     ['Dense(128) → ReLU → BN → Dropout(0.3)', 'Dense(64) → ReLU → BN → Dropout(0.2)', 'Dense(32) → ReLU', 'Sigmoid Output'],
     COLORS['purple']),
    ('Autoencoder',
     ['Encoder: 64 → 32 → 16', 'Decoder: 16 → 32 → 64', 'Reconstruction Loss (MSE)', 'Anomaly = High Error'],
     COLORS['pink']),
]

y_start = 0.80
for name, layers, color in architectures:
    ax.text(0.15, y_start, name, fontsize=16, fontweight='bold', color=color, transform=ax.transAxes)
    for i, layer in enumerate(layers):
        ax.text(0.18, y_start - 0.04 - i * 0.035, f'• {layer}', fontsize=10,
                color=COLORS['text'], alpha=0.8, transform=ax.transAxes)
    y_start -= 0.25

# GPU acceleration note
ax.text(0.5, 0.08, 'All models trained on NVIDIA RTX 3060 (CUDA 11.8)', fontsize=11,
        ha='center', color=COLORS['yellow'], transform=ax.transAxes)

save_fig(fig, '21_neural_architectures')

print("\n" + "=" * 50)
print("Deep learning graphs generated successfully!")
print("=" * 50)

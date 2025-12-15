"""
DEEP LEARNING MOBILE GRAPHS
===========================
Mobile-optimized graphs for DL results (600x800px portrait)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# Mobile style
M = {
    'figsize': (6, 8),
    'bg': '#0d1117',
    'card': '#161b22',
    'text': '#f0f6fc',
    'text_dim': '#8b949e',
    'accent': '#58a6ff',
    'green': '#3fb950',
    'red': '#f85149',
    'yellow': '#d29922',
    'purple': '#a371f7',
    'orange': '#db6d28',
    'cyan': '#39c5cf',
    'pink': '#ff7b72',
    'dpi': 200
}

def save_mobile(fig, name):
    fig.savefig(f'graphs_mobile/{name}.png', dpi=M['dpi'], bbox_inches='tight',
                facecolor=M['bg'], edgecolor='none', pad_inches=0.3)
    plt.close(fig)
    print(f"   Saved: graphs_mobile/{name}.png")

print("Loading results...")
dl_results = joblib.load('models/deep_learning_results.joblib')
ml_results = joblib.load('models/results_summary.joblib')

print("Generating deep learning mobile graphs...")

# ============================================================================
# 12. DL VS ML COMPARISON
# ============================================================================
print("\n[12] DL vs ML Comparison...")
fig, ax = plt.subplots(figsize=M['figsize'], facecolor=M['bg'])
ax.set_facecolor(M['bg'])
ax.axis('off')

ax.text(0.5, 0.95, 'DEEP LEARNING', fontsize=24, fontweight='bold',
        ha='center', color=M['text'], transform=ax.transAxes)
ax.text(0.5, 0.88, 'vs TRADITIONAL ML', fontsize=18,
        ha='center', color=M['accent'], transform=ax.transAxes)

ax.text(0.5, 0.74, 'BEST MODEL', fontsize=14, fontweight='bold',
        ha='center', color=M['yellow'], transform=ax.transAxes)
ax.text(0.5, 0.62, '1D-CNN', fontsize=36, fontweight='bold',
        ha='center', color=M['cyan'], transform=ax.transAxes)
ax.text(0.5, 0.52, '93.69%', fontsize=56, fontweight='bold',
        ha='center', color=M['green'], transform=ax.transAxes)
ax.text(0.5, 0.44, 'ACCURACY', fontsize=14, ha='center',
        color=M['text_dim'], transform=ax.transAxes)

# Comparison
models = [
    ('DNN', '93.47%', M['purple']),
    ('XGBoost', '90.04%', M['green']),
    ('LightGBM', '89.93%', M['accent']),
    ('Random Forest', '89.99%', M['orange']),
]

for i, (name, acc, color) in enumerate(models):
    y = 0.32 - i * 0.07
    ax.text(0.2, y, name, fontsize=12, ha='left',
            color=M['text'], transform=ax.transAxes)
    ax.text(0.8, y, acc, fontsize=12, fontweight='bold', ha='right',
            color=color, transform=ax.transAxes)

ax.text(0.5, 0.05, '+3.65% over XGBoost', fontsize=11, ha='center',
        color=M['yellow'], transform=ax.transAxes)

save_mobile(fig, '12_dl_vs_ml')

# ============================================================================
# 13. MODEL LEADERBOARD
# ============================================================================
print("[13] Model Leaderboard...")
fig, ax = plt.subplots(figsize=M['figsize'], facecolor=M['bg'])
ax.set_facecolor(M['bg'])
ax.axis('off')

ax.text(0.5, 0.95, 'MODEL LEADERBOARD', fontsize=22, fontweight='bold',
        ha='center', color=M['accent'], transform=ax.transAxes)

leaderboard = [
    ('🥇', '1D-CNN', '93.69%', M['cyan']),
    ('🥈', 'DNN', '93.47%', M['purple']),
    ('🥉', 'XGBoost', '90.04%', M['green']),
    ('4', 'Random Forest', '89.99%', M['orange']),
    ('5', 'LightGBM', '89.93%', M['accent']),
    ('6', 'Ensemble Stack', '~90%', M['yellow']),
    ('7', 'Isolation Forest', '~80%', M['pink']),
    ('8', 'Autoencoder', '69.12%', M['pink']),
]

for i, (rank, name, acc, color) in enumerate(leaderboard):
    y = 0.82 - i * 0.09
    ax.text(0.08, y, rank, fontsize=12, ha='left',
            color=M['text'], transform=ax.transAxes)
    ax.text(0.18, y, name, fontsize=12, fontweight='bold', ha='left',
            color=color, transform=ax.transAxes)
    ax.text(0.92, y, acc, fontsize=12, ha='right',
            color=M['text'], transform=ax.transAxes)

ax.text(0.5, 0.03, '8 Models Trained & Evaluated', fontsize=10,
        ha='center', color=M['text_dim'], transform=ax.transAxes)

save_mobile(fig, '13_leaderboard')

# ============================================================================
# 14. GPU ACCELERATION
# ============================================================================
print("[14] GPU Acceleration...")
fig, ax = plt.subplots(figsize=M['figsize'], facecolor=M['bg'])
ax.set_facecolor(M['bg'])
ax.axis('off')

ax.text(0.5, 0.92, 'GPU ACCELERATION', fontsize=22, fontweight='bold',
        ha='center', color=M['green'], transform=ax.transAxes)

ax.text(0.5, 0.75, 'RTX 3060', fontsize=42, fontweight='bold',
        ha='center', color=M['cyan'], transform=ax.transAxes)
ax.text(0.5, 0.65, 'NVIDIA CUDA 11.8', fontsize=14,
        ha='center', color=M['text_dim'], transform=ax.transAxes)

# Architectures
ax.text(0.5, 0.50, 'NEURAL NETWORKS', fontsize=14, fontweight='bold',
        ha='center', color=M['accent'], transform=ax.transAxes)

archs = [
    ('1D-CNN', 'Conv → Pool → Dense'),
    ('DNN', '128 → 64 → 32 → 1'),
    ('Autoencoder', 'Encoder → Decoder'),
]

for i, (name, desc) in enumerate(archs):
    y = 0.42 - i * 0.10
    ax.text(0.15, y, name, fontsize=12, fontweight='bold', ha='left',
            color=M['purple'], transform=ax.transAxes)
    ax.text(0.15, y - 0.04, desc, fontsize=9, ha='left',
            color=M['text_dim'], transform=ax.transAxes)

ax.text(0.5, 0.08, 'PyTorch 2.7.1', fontsize=11,
        ha='center', color=M['yellow'], transform=ax.transAxes)

save_mobile(fig, '14_gpu_acceleration')

# ============================================================================
# 15. FINAL SUMMARY
# ============================================================================
print("[15] Final Summary...")
fig, ax = plt.subplots(figsize=M['figsize'], facecolor=M['bg'])
ax.set_facecolor(M['bg'])
ax.axis('off')

ax.text(0.5, 0.95, 'PROJECT SUMMARY', fontsize=24, fontweight='bold',
        ha='center', color=M['accent'], transform=ax.transAxes)

highlights = [
    ('93.7%', 'Best Accuracy', M['green']),
    ('8', 'ML Models', M['accent']),
    ('3M+', 'Records', M['purple']),
    ('3', 'Datasets', M['cyan']),
    ('21', 'Visualizations', M['orange']),
    ('9', 'Attack Types', M['red']),
]

for i, (value, label, color) in enumerate(highlights):
    y = 0.78 - i * 0.12
    ax.text(0.15, y, value, fontsize=32, fontweight='bold',
            ha='left', color=color, transform=ax.transAxes)
    ax.text(0.48, y, label, fontsize=14, ha='left', va='center',
            color=M['text'], transform=ax.transAxes)

ax.text(0.5, 0.05, 'Network Intrusion Detection | Multi-Source AI', fontsize=10,
        ha='center', color=M['text_dim'], transform=ax.transAxes)

save_mobile(fig, '15_final_summary')

print("\n" + "=" * 50)
print("Deep learning mobile graphs generated!")
print("=" * 50)

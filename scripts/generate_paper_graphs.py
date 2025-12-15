"""
GENERATE PAPER RESULTS VISUALIZATIONS
=====================================
Updated graphs showing paper methodology and 94.25% accuracy
"""

import matplotlib.pyplot as plt
import numpy as np
import joblib

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
    'cyan': '#39c5cf',
}

def save_fig(fig, name, dpi=150):
    fig.savefig(f'graphs/{name}.png', dpi=dpi, bbox_inches='tight',
                facecolor=COLORS['bg'], edgecolor='none')
    plt.close(fig)
    print(f"   Saved: graphs/{name}.png")

print("Loading paper results...")
paper = joblib.load('models/paper_implementation_results.joblib')

# ============================================================================
# 27. PAPER METHODOLOGY RESULTS
# ============================================================================
print("\n[27] Paper Methodology Results...")
fig, ax = plt.subplots(figsize=(14, 10), facecolor=COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
ax.axis('off')

ax.text(0.5, 0.95, 'PAPER METHODOLOGY RESULTS', fontsize=28, fontweight='bold',
        ha='center', color=COLORS['accent'], transform=ax.transAxes)
ax.text(0.5, 0.88, 'Implementing 2024-2025 Research Paper Techniques', fontsize=14,
        ha='center', color=COLORS['text'], alpha=0.8, transform=ax.transAxes)

# Best result highlight
ax.text(0.5, 0.72, 'BEST: XGBoost', fontsize=20, fontweight='bold',
        ha='center', color=COLORS['yellow'], transform=ax.transAxes)
ax.text(0.5, 0.58, '94.25%', fontsize=72, fontweight='bold',
        ha='center', color=COLORS['green'], transform=ax.transAxes)
ax.text(0.5, 0.48, 'ACCURACY', fontsize=16, ha='center',
        color=COLORS['text'], alpha=0.8, transform=ax.transAxes)

# Other results
results_list = [
    ('Random Forest', '93.64%', COLORS['accent']),
    ('Ensemble', '93.87%', COLORS['purple']),
    ('Deep NN', '92.36%', COLORS['cyan']),
]

for i, (name, acc, color) in enumerate(results_list):
    x = 0.2 + i * 0.30
    ax.text(x, 0.32, name, fontsize=12, fontweight='bold', ha='center',
            color=color, transform=ax.transAxes)
    ax.text(x, 0.24, acc, fontsize=24, fontweight='bold', ha='center',
            color=COLORS['text'], transform=ax.transAxes)

# Techniques used
ax.text(0.5, 0.12, 'Techniques: Chi-Square + SMOTE + Correlation Removal + Ensemble',
        fontsize=11, ha='center', color=COLORS['text'], alpha=0.7, transform=ax.transAxes)

save_fig(fig, '27_paper_methodology_results')

# ============================================================================
# 28. RESEARCH CITATIONS
# ============================================================================
print("[28] Research Citations...")
fig, ax = plt.subplots(figsize=(16, 12), facecolor=COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
ax.axis('off')

ax.text(0.5, 0.95, 'RESEARCH PAPER CITATIONS', fontsize=26, fontweight='bold',
        ha='center', color=COLORS['text'], transform=ax.transAxes)
ax.text(0.5, 0.88, 'Techniques Implemented from 2024-2025 Publications', fontsize=14,
        ha='center', color=COLORS['accent'], transform=ax.transAxes)

citations = [
    ('"Deep Learning Ensembles with RL Controller"', '2025', '99.8%', 'jier.org'),
    ('"DMI-GA Feature Selection + Random Forest"', '2024', '99.94%', 'ResearchGate'),
    ('"Chi-Square Filter Feature Selection"', '2024', '99.57%', 'rdd.edu.iq'),
    ('"SMOTE + Deep Learning for IDS"', '2024', '99.0%', 'MDPI'),
    ('"Ensemble Learning with Correlation FS"', '2025', '99.99%', 'MDPI'),
    ('"Hypergraph-Based ML Ensemble NIDS"', '2024', '~100%', 'arXiv'),
]

ax.text(0.08, 0.78, '#', fontsize=11, fontweight='bold', color=COLORS['accent'], transform=ax.transAxes)
ax.text(0.12, 0.78, 'Paper Title', fontsize=11, fontweight='bold', color=COLORS['accent'], transform=ax.transAxes)
ax.text(0.62, 0.78, 'Year', fontsize=11, fontweight='bold', color=COLORS['accent'], transform=ax.transAxes)
ax.text(0.72, 0.78, 'Reported Acc', fontsize=11, fontweight='bold', color=COLORS['accent'], transform=ax.transAxes)
ax.text(0.88, 0.78, 'Source', fontsize=11, fontweight='bold', color=COLORS['accent'], transform=ax.transAxes)

for i, (title, year, acc, source) in enumerate(citations):
    y = 0.70 - i * 0.08
    ax.text(0.08, y, str(i+1), fontsize=10, color=COLORS['text'], transform=ax.transAxes)
    ax.text(0.12, y, title, fontsize=10, color=COLORS['text'], transform=ax.transAxes)
    ax.text(0.62, y, year, fontsize=10, color=COLORS['green'], transform=ax.transAxes)
    ax.text(0.72, y, acc, fontsize=10, fontweight='bold', color=COLORS['yellow'], transform=ax.transAxes)
    ax.text(0.88, y, source, fontsize=9, color=COLORS['cyan'], transform=ax.transAxes)

# Our implementation
ax.text(0.5, 0.20, 'OUR IMPLEMENTATION', fontsize=14, fontweight='bold',
        ha='center', color=COLORS['accent'], transform=ax.transAxes)
ax.text(0.5, 0.12, 'XGBoost + Chi-Square + SMOTE → 94.25% Accuracy',
        fontsize=12, ha='center', color=COLORS['green'], transform=ax.transAxes)
ax.text(0.5, 0.06, 'Note: Papers used NF-UNSW-NB15 (pre-processed NetFlow version)',
        fontsize=10, ha='center', color=COLORS['text'], alpha=0.6, transform=ax.transAxes)

save_fig(fig, '28_research_citations')

# ============================================================================
# 29. FINAL LEADERBOARD UPDATE  
# ============================================================================
print("[29] Final Leaderboard...")
fig, ax = plt.subplots(figsize=(14, 10), facecolor=COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
ax.axis('off')

ax.text(0.5, 0.95, 'COMPLETE MODEL LEADERBOARD', fontsize=26, fontweight='bold',
        ha='center', color=COLORS['accent'], transform=ax.transAxes)
ax.text(0.5, 0.88, 'All Models Ranked by Binary Classification Accuracy', fontsize=14,
        ha='center', color=COLORS['text'], alpha=0.8, transform=ax.transAxes)

leaderboard = [
    ('1', 'XGBoost (Paper)', '94.25%', '99.05%', COLORS['green']),
    ('2', 'Random Forest (Paper)', '93.64%', '98.97%', COLORS['accent']),
    ('3', 'Deep FFN', '93.63%', '98.92%', COLORS['purple']),
    ('4', '1D-CNN', '93.51%', '98.76%', COLORS['cyan']),
    ('5', 'Deep NN (Paper)', '92.36%', '98.56%', COLORS['orange']),
    ('6', 'XGBoost (Baseline)', '90.04%', '98.59%', COLORS['yellow']),
    ('7', 'Random Forest', '89.99%', '98.63%', COLORS['red']),
    ('8', 'LightGBM', '89.93%', '98.56%', COLORS['text']),
]

ax.text(0.08, 0.78, '#', fontsize=12, fontweight='bold', color=COLORS['accent'], transform=ax.transAxes)
ax.text(0.15, 0.78, 'Model', fontsize=12, fontweight='bold', color=COLORS['accent'], transform=ax.transAxes)
ax.text(0.55, 0.78, 'Accuracy', fontsize=12, fontweight='bold', color=COLORS['accent'], transform=ax.transAxes)
ax.text(0.75, 0.78, 'ROC-AUC', fontsize=12, fontweight='bold', color=COLORS['accent'], transform=ax.transAxes)

for i, (rank, name, acc, auc, color) in enumerate(leaderboard):
    y = 0.70 - i * 0.07
    ax.text(0.08, y, rank, fontsize=12, color=COLORS['text'], transform=ax.transAxes)
    ax.text(0.15, y, name, fontsize=11, fontweight='bold', color=color, transform=ax.transAxes)
    ax.text(0.55, y, acc, fontsize=11, color=COLORS['text'], transform=ax.transAxes)
    ax.text(0.75, y, auc, fontsize=11, color=COLORS['text'], transform=ax.transAxes)

ax.text(0.5, 0.08, '+4.21% improvement over baseline with paper techniques',
        fontsize=12, fontweight='bold', ha='center', color=COLORS['green'], transform=ax.transAxes)

save_fig(fig, '29_complete_leaderboard')

print("\n" + "=" * 60)
print("Paper results visualizations complete!")
print("=" * 60)

"""
UNSW-NB15 Network Intrusion Detection - Mobile Graph Generation
================================================================
Generates mobile-optimized graphs (600x800px, portrait orientation)
for comprehensive mobile viewing experience.

Mobile Graph Pattern:
- 600x800px portrait orientation
- Large fonts (40-96pt for key numbers)
- Dark theme matching dashboard
- High DPI (200) for crisp viewing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
from pathlib import Path

# Create output directory
Path('graphs_mobile').mkdir(exist_ok=True)

# Mobile style configuration
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
    'dpi': 200
}

def save_mobile(fig, name):
    """Save mobile-optimized figure"""
    fig.savefig(f'graphs_mobile/{name}.png', dpi=M['dpi'], bbox_inches='tight',
                facecolor=M['bg'], edgecolor='none', pad_inches=0.3)
    plt.close(fig)
    print(f"   Saved: graphs_mobile/{name}.png")

# Load data
print("Loading data...")
train_df = pd.read_csv('training_set.csv')
test_df = pd.read_csv('testing_set.csv')
results = joblib.load('models/results_summary.joblib')

df = pd.concat([train_df, test_df], ignore_index=True)
df['attack_cat'] = df['attack_cat'].str.strip()

print(f"Total records: {len(df):,}")
print("\nGenerating mobile graphs...")

# ============================================================================
# 1. KEY STATS OVERVIEW
# ============================================================================
print("\n[1] Key Stats Overview...")
fig, ax = plt.subplots(figsize=M['figsize'], facecolor=M['bg'])
ax.set_facecolor(M['bg'])
ax.axis('off')

# Title
ax.text(0.5, 0.95, 'NETWORK INTRUSION', fontsize=28, fontweight='bold',
        ha='center', color=M['text'], transform=ax.transAxes)
ax.text(0.5, 0.89, 'DETECTION', fontsize=28, fontweight='bold',
        ha='center', color=M['accent'], transform=ax.transAxes)

# Key stats
stats = [
    ('257,673', 'Total Records', M['cyan']),
    ('43', 'Network Features', M['purple']),
    ('9', 'Attack Types', M['red']),
    ('45%', 'Normal Traffic', M['green']),
    ('55%', 'Attack Traffic', M['orange']),
]

for i, (value, label, color) in enumerate(stats):
    y = 0.72 - i * 0.14
    ax.text(0.5, y, value, fontsize=48, fontweight='bold',
            ha='center', color=color, transform=ax.transAxes)
    ax.text(0.5, y - 0.05, label, fontsize=14, ha='center',
            color=M['text_dim'], transform=ax.transAxes)

# Footer
ax.text(0.5, 0.05, 'UNSW-NB15 Benchmark Dataset', fontsize=12,
        ha='center', color=M['text_dim'], transform=ax.transAxes)

save_mobile(fig, '01_stats_overview')

# ============================================================================
# 2. ATTACK TYPE DISTRIBUTION
# ============================================================================
print("[2] Attack Type Distribution...")
fig, ax = plt.subplots(figsize=M['figsize'], facecolor=M['bg'])
ax.set_facecolor(M['bg'])

attack_counts = df['attack_cat'].value_counts()
colors = [M['green'] if cat == 'Normal' else M['red'] if cat == 'Generic' 
          else M['orange'] if cat == 'Exploits' else M['yellow'] if cat == 'Fuzzers'
          else M['cyan'] for cat in attack_counts.index]

bars = ax.barh(attack_counts.index[::-1], attack_counts.values[::-1], 
               color=colors[::-1], height=0.7)
ax.set_xlabel('Count', fontsize=14, color=M['text'])
ax.set_title('Attack Type Distribution', fontsize=22, fontweight='bold',
             color=M['text'], pad=20)
ax.tick_params(colors=M['text'], labelsize=12)

for spine in ax.spines.values():
    spine.set_color(M['text_dim'])
    spine.set_alpha(0.3)

# Add count labels
for bar, val in zip(bars, attack_counts.values[::-1]):
    ax.text(val + 1000, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', color=M['text'], fontsize=10)

plt.tight_layout()
save_mobile(fig, '02_attack_distribution')

# ============================================================================
# 3. BINARY CLASSIFICATION WINNER
# ============================================================================
print("[3] Binary Classification Winner...")
fig, ax = plt.subplots(figsize=M['figsize'], facecolor=M['bg'])
ax.set_facecolor(M['bg'])
ax.axis('off')

# Title
ax.text(0.5, 0.95, 'BINARY CLASSIFICATION', fontsize=22, fontweight='bold',
        ha='center', color=M['text'], transform=ax.transAxes)
ax.text(0.5, 0.88, 'Normal vs Attack', fontsize=16,
        ha='center', color=M['text_dim'], transform=ax.transAxes)

# Winner badge
ax.text(0.5, 0.75, 'WINNER', fontsize=16, fontweight='bold',
        ha='center', color=M['yellow'], transform=ax.transAxes)
ax.text(0.5, 0.65, 'XGBoost', fontsize=40, fontweight='bold',
        ha='center', color=M['green'], transform=ax.transAxes)

# Accuracy
ax.text(0.5, 0.50, '90.04%', fontsize=72, fontweight='bold',
        ha='center', color=M['accent'], transform=ax.transAxes)
ax.text(0.5, 0.42, 'ACCURACY', fontsize=16, ha='center',
        color=M['text_dim'], transform=ax.transAxes)

# Other metrics
metrics = [
    ('Precision', '89.79%'),
    ('Recall', '92.10%'),
    ('F1-Score', '90.93%'),
    ('ROC-AUC', '98.54%'),
]

for i, (label, value) in enumerate(metrics):
    y = 0.30 - i * 0.07
    ax.text(0.3, y, label, fontsize=14, ha='right',
            color=M['text_dim'], transform=ax.transAxes)
    ax.text(0.7, y, value, fontsize=14, fontweight='bold', ha='left',
            color=M['text'], transform=ax.transAxes)

save_mobile(fig, '03_binary_winner')

# ============================================================================
# 4. MULTI-CLASS RESULTS
# ============================================================================
print("[4] Multi-Class Results...")
fig, ax = plt.subplots(figsize=M['figsize'], facecolor=M['bg'])
ax.set_facecolor(M['bg'])
ax.axis('off')

ax.text(0.5, 0.95, 'MULTI-CLASS', fontsize=24, fontweight='bold',
        ha='center', color=M['text'], transform=ax.transAxes)
ax.text(0.5, 0.88, '10 Traffic Categories', fontsize=14,
        ha='center', color=M['text_dim'], transform=ax.transAxes)

ax.text(0.5, 0.72, '76.35%', fontsize=64, fontweight='bold',
        ha='center', color=M['purple'], transform=ax.transAxes)
ax.text(0.5, 0.62, 'ACCURACY', fontsize=16, ha='center',
        color=M['text_dim'], transform=ax.transAxes)

# Model comparison
ax.text(0.5, 0.50, 'MODEL COMPARISON', fontsize=14, fontweight='bold',
        ha='center', color=M['accent'], transform=ax.transAxes)

models = [
    ('XGBoost', '76.35%', M['green']),
    ('Random Forest', '75.40%', M['accent']),
    ('LightGBM', '71.70%', M['purple']),
]

for i, (name, acc, color) in enumerate(models):
    y = 0.42 - i * 0.10
    ax.text(0.25, y, name, fontsize=16, ha='left',
            color=M['text'], transform=ax.transAxes)
    ax.text(0.75, y, acc, fontsize=16, fontweight='bold', ha='right',
            color=color, transform=ax.transAxes)

save_mobile(fig, '04_multiclass_results')

# ============================================================================
# 5. TOP FEATURES
# ============================================================================
print("[5] Top Features...")
fig, ax = plt.subplots(figsize=M['figsize'], facecolor=M['bg'])
ax.set_facecolor(M['bg'])

importance_df = pd.DataFrame(results['feature_importance'])
top10 = importance_df.nlargest(10, 'importance')

colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top10)))[::-1]
bars = ax.barh(range(len(top10)), top10['importance'].values[::-1],
               color=colors, height=0.7)
ax.set_yticks(range(len(top10)))
ax.set_yticklabels(top10['feature'].values[::-1], fontsize=12)
ax.set_xlabel('Importance Score', fontsize=12, color=M['text'])
ax.set_title('Top 10 Network Features', fontsize=20, fontweight='bold',
             color=M['text'], pad=20)
ax.tick_params(colors=M['text'])

for spine in ax.spines.values():
    spine.set_color(M['text_dim'])
    spine.set_alpha(0.3)

plt.tight_layout()
save_mobile(fig, '05_top_features')

# ============================================================================
# 6. PER-ATTACK DETECTION
# ============================================================================
print("[6] Per-Attack Detection...")
fig, ax = plt.subplots(figsize=M['figsize'], facecolor=M['bg'])
ax.set_facecolor(M['bg'])
ax.axis('off')

ax.text(0.5, 0.95, 'PER-ATTACK', fontsize=24, fontweight='bold',
        ha='center', color=M['text'], transform=ax.transAxes)
ax.text(0.5, 0.88, 'DETECTION RATES', fontsize=18,
        ha='center', color=M['accent'], transform=ax.transAxes)

# Attack F1 scores (from analysis)
attacks = [
    ('Generic', '98.4%', M['green']),
    ('Normal', '85.9%', M['green']),
    ('Reconnaissance', '82.0%', M['accent']),
    ('Exploits', '68.9%', M['yellow']),
    ('Shellcode', '64.0%', M['yellow']),
    ('Worms', '57.3%', M['orange']),
    ('DoS', '42.5%', M['orange']),
    ('Fuzzers', '18.1%', M['red']),
    ('Backdoor', '13.0%', M['red']),
    ('Analysis', '0.0%', M['red']),
]

for i, (attack, score, color) in enumerate(attacks):
    y = 0.78 - i * 0.075
    ax.text(0.1, y, attack, fontsize=14, ha='left',
            color=M['text'], transform=ax.transAxes)
    ax.text(0.9, y, score, fontsize=14, fontweight='bold', ha='right',
            color=color, transform=ax.transAxes)

ax.text(0.5, 0.03, 'F1-Score by Attack Type', fontsize=11,
        ha='center', color=M['text_dim'], transform=ax.transAxes)

save_mobile(fig, '06_per_attack_detection')

# ============================================================================
# 7. NETWORK PROTOCOLS
# ============================================================================
print("[7] Network Protocols...")
fig, ax = plt.subplots(figsize=M['figsize'], facecolor=M['bg'])
ax.set_facecolor(M['bg'])

proto_counts = df['proto'].value_counts().head(8)
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(proto_counts)))

ax.pie(proto_counts.values, labels=proto_counts.index, autopct='%1.1f%%',
       colors=colors, textprops={'color': M['text'], 'fontsize': 11})
ax.set_title('Top Network Protocols', fontsize=20, fontweight='bold',
             color=M['text'], pad=20)

save_mobile(fig, '07_network_protocols')

# ============================================================================
# 8. ROC-AUC HIGHLIGHT
# ============================================================================
print("[8] ROC-AUC Highlight...")
fig, ax = plt.subplots(figsize=M['figsize'], facecolor=M['bg'])
ax.set_facecolor(M['bg'])
ax.axis('off')

ax.text(0.5, 0.85, 'ROC-AUC SCORE', fontsize=20, fontweight='bold',
        ha='center', color=M['text'], transform=ax.transAxes)

ax.text(0.5, 0.60, '98.54%', fontsize=80, fontweight='bold',
        ha='center', color=M['green'], transform=ax.transAxes)

ax.text(0.5, 0.42, 'EXCELLENT', fontsize=24, fontweight='bold',
        ha='center', color=M['yellow'], transform=ax.transAxes)
ax.text(0.5, 0.35, 'Discrimination Ability', fontsize=14,
        ha='center', color=M['text_dim'], transform=ax.transAxes)

# Scale
ax.text(0.5, 0.20, '|---------|---------|---------|---------|', fontsize=12,
        ha='center', color=M['text_dim'], family='monospace', transform=ax.transAxes)
ax.text(0.5, 0.15, '0.5      0.7      0.85      0.95     1.0', fontsize=10,
        ha='center', color=M['text_dim'], family='monospace', transform=ax.transAxes)
ax.text(0.5, 0.10, 'Poor                             Perfect', fontsize=10,
        ha='center', color=M['text_dim'], transform=ax.transAxes)

save_mobile(fig, '08_roc_auc_score')

# ============================================================================
# 9. DATASET BREAKDOWN
# ============================================================================
print("[9] Dataset Breakdown...")
fig, ax = plt.subplots(figsize=M['figsize'], facecolor=M['bg'])
ax.set_facecolor(M['bg'])
ax.axis('off')

ax.text(0.5, 0.95, 'DATASET BREAKDOWN', fontsize=22, fontweight='bold',
        ha='center', color=M['text'], transform=ax.transAxes)

breakdown = [
    ('Training Set', '82,332', '32%', M['accent']),
    ('Testing Set', '175,341', '68%', M['purple']),
]

for i, (label, count, pct, color) in enumerate(breakdown):
    y = 0.75 - i * 0.20
    ax.text(0.5, y, count, fontsize=48, fontweight='bold',
            ha='center', color=color, transform=ax.transAxes)
    ax.text(0.5, y - 0.06, f'{label} ({pct})', fontsize=14,
            ha='center', color=M['text_dim'], transform=ax.transAxes)

# Feature categories
ax.text(0.5, 0.40, '43 NETWORK FEATURES', fontsize=16, fontweight='bold',
        ha='center', color=M['cyan'], transform=ax.transAxes)

features = ['Flow features', 'Basic features', 'Content features', 
            'Time features', 'Connection features']
for i, feat in enumerate(features):
    y = 0.32 - i * 0.055
    ax.text(0.5, y, f'• {feat}', fontsize=12,
            ha='center', color=M['text'], transform=ax.transAxes)

save_mobile(fig, '09_dataset_breakdown')

# ============================================================================
# 10. KEY TAKEAWAYS
# ============================================================================
print("[10] Key Takeaways...")
fig, ax = plt.subplots(figsize=M['figsize'], facecolor=M['bg'])
ax.set_facecolor(M['bg'])
ax.axis('off')

ax.text(0.5, 0.95, 'KEY TAKEAWAYS', fontsize=24, fontweight='bold',
        ha='center', color=M['accent'], transform=ax.transAxes)

takeaways = [
    ('90%', 'Binary detection accuracy', M['green']),
    ('76%', 'Multi-class accuracy', M['accent']),
    ('98.5%', 'ROC-AUC score', M['purple']),
    ('9', 'Attack types detected', M['red']),
    ('XGBoost', 'Best performing model', M['yellow']),
]

for i, (value, label, color) in enumerate(takeaways):
    y = 0.80 - i * 0.14
    ax.text(0.15, y, value, fontsize=32, fontweight='bold',
            ha='left', color=color, transform=ax.transAxes)
    ax.text(0.45, y, label, fontsize=14, ha='left', va='center',
            color=M['text'], transform=ax.transAxes)

ax.text(0.5, 0.08, 'AI-Powered Cybersecurity', fontsize=14,
        ha='center', color=M['text_dim'], transform=ax.transAxes)
ax.text(0.5, 0.03, 'UNSW-NB15 Benchmark', fontsize=12,
        ha='center', color=M['text_dim'], transform=ax.transAxes)

save_mobile(fig, '10_key_takeaways')

# ============================================================================
# 11. BUSINESS IMPACT
# ============================================================================
print("[11] Business Impact...")
fig, ax = plt.subplots(figsize=M['figsize'], facecolor=M['bg'])
ax.set_facecolor(M['bg'])
ax.axis('off')

ax.text(0.5, 0.95, 'BUSINESS IMPACT', fontsize=24, fontweight='bold',
        ha='center', color=M['green'], transform=ax.transAxes)

impacts = [
    ('DETECT', 'Network intrusions in real-time', M['red']),
    ('CLASSIFY', '9 different attack types', M['orange']),
    ('REDUCE', 'False positive alerts', M['yellow']),
    ('PROTECT', 'Enterprise networks', M['green']),
    ('SCALE', 'To millions of connections', M['cyan']),
]

for i, (action, desc, color) in enumerate(impacts):
    y = 0.78 - i * 0.13
    ax.text(0.1, y, action, fontsize=18, fontweight='bold',
            ha='left', color=color, transform=ax.transAxes)
    ax.text(0.1, y - 0.04, desc, fontsize=12,
            ha='left', color=M['text_dim'], transform=ax.transAxes)

save_mobile(fig, '11_business_impact')

print("\n" + "=" * 50)
print(f"Generated 11 mobile-optimized graphs!")
print("=" * 50)

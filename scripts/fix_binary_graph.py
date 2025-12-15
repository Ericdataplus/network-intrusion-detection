"""
Fix for binary classification graph - better aspect ratio to fill card
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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
}

# Load data
results = joblib.load('models/results_summary.joblib')
cm_binary = np.load('models/confusion_matrix_binary.npy')

print("Regenerating binary classification graph with better aspect ratio...")

# FIXED: Use better aspect ratio (14x10 instead of 18x6)
fig = plt.figure(figsize=(14, 10), facecolor=COLORS['bg'])

# Grid layout: 2 rows
# Top row: metrics bar chart | confusion matrix
# Bottom row: best model summary (centered, full width)

# Top left - Metrics bar chart
ax1 = fig.add_axes([0.05, 0.45, 0.42, 0.48])
ax1.set_facecolor(COLORS['bg'])

binary_results = results['binary_results']
models = list(binary_results.keys())
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']

x = np.arange(len(models))
width = 0.15
colors_metrics = [COLORS['green'], COLORS['accent'], COLORS['purple'], COLORS['orange'], COLORS['yellow']]

for i, (metric, color) in enumerate(zip(metrics, colors_metrics)):
    values = [binary_results[m][metric] for m in models]
    ax1.bar(x + i*width, values, width, label=metric_labels[i], color=color, alpha=0.9)

ax1.set_ylabel('Score', fontsize=12, color=COLORS['text'])
ax1.set_title('Binary Classification Metrics', fontsize=16, fontweight='bold', color=COLORS['text'])
ax1.set_xticks(x + width * 2)
ax1.set_xticklabels(models, color=COLORS['text'], fontsize=11)
ax1.tick_params(colors=COLORS['text'])
ax1.legend(loc='lower right', fontsize=9, ncol=2)
ax1.set_ylim(0.85, 1.02)
ax1.axhline(y=0.9, color=COLORS['yellow'], linestyle='--', alpha=0.5)
for spine in ax1.spines.values():
    spine.set_color(COLORS['text'])
    spine.set_alpha(0.3)

# Top right - Confusion matrix
ax2 = fig.add_axes([0.54, 0.45, 0.42, 0.48])
ax2.set_facecolor(COLORS['bg'])
sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'],
            cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
ax2.set_title('Confusion Matrix (XGBoost)', fontsize=16, fontweight='bold', color=COLORS['text'])
ax2.set_xlabel('Predicted', color=COLORS['text'], fontsize=12)
ax2.set_ylabel('Actual', color=COLORS['text'], fontsize=12)
ax2.tick_params(colors=COLORS['text'])

# Bottom - Best model summary (full width, centered)
ax3 = fig.add_axes([0.15, 0.05, 0.70, 0.32])
ax3.set_facecolor(COLORS['card'])
ax3.axis('off')

best_model = 'XGBoost'
best_metrics = binary_results[best_model]

# Title
ax3.text(0.5, 0.90, 'BEST MODEL', fontsize=14, ha='center', 
         color=COLORS['accent'], fontweight='bold', transform=ax3.transAxes)
ax3.text(0.5, 0.72, best_model, fontsize=32, ha='center',
         color=COLORS['green'], fontweight='bold', transform=ax3.transAxes)

# Metrics in a horizontal row
metric_data = [
    ('Accuracy', f"{best_metrics['accuracy']*100:.2f}%"),
    ('Precision', f"{best_metrics['precision']*100:.2f}%"),
    ('Recall', f"{best_metrics['recall']*100:.2f}%"),
    ('F1 Score', f"{best_metrics['f1']*100:.2f}%"),
    ('ROC-AUC', f"{best_metrics['roc_auc']:.4f}"),
]

for i, (label, value) in enumerate(metric_data):
    x_pos = 0.1 + i * 0.18
    ax3.text(x_pos, 0.42, value, fontsize=18, fontweight='bold',
             color=COLORS['text'], transform=ax3.transAxes)
    ax3.text(x_pos, 0.20, label, fontsize=10, 
             color=COLORS['text'], alpha=0.7, transform=ax3.transAxes)

# Add border
for spine in ax3.spines.values():
    spine.set_visible(True)
    spine.set_color(COLORS['accent'])
    spine.set_linewidth(2)

fig.savefig('graphs/03_binary_classification.png', dpi=150, bbox_inches='tight',
            facecolor=COLORS['bg'], edgecolor='none')
plt.close(fig)

print("Fixed! Saved: graphs/03_binary_classification.png")

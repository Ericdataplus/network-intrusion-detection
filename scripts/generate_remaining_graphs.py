"""
Generate remaining graphs (7-10) for the dashboard
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
    'pink': '#ff7b72',
    'cyan': '#39c5cf',
}

ATTACK_COLORS = {
    'Normal': COLORS['green'],
    'Generic': COLORS['red'],
    'Exploits': COLORS['orange'],
    'Fuzzers': COLORS['yellow'],
    'DoS': COLORS['pink'],
    'Reconnaissance': COLORS['purple'],
    'Backdoor': COLORS['cyan'],
    'Analysis': COLORS['accent'],
    'Shellcode': '#ff9f43',
    'Worms': '#ee5253',
}

def save_fig(fig, name, dpi=150):
    fig.savefig(f'graphs/{name}.png', dpi=dpi, bbox_inches='tight', 
                facecolor=COLORS['bg'], edgecolor='none')
    plt.close(fig)
    print(f"   Saved: graphs/{name}.png")

# Load data
print("Loading data...")
train_df = pd.read_csv('training_set.csv')
test_df = pd.read_csv('testing_set.csv')
results = joblib.load('models/results_summary.joblib')

# Combine datasets
df = pd.concat([train_df, test_df], ignore_index=True)
df['attack_cat'] = df['attack_cat'].str.strip()

# Load models  
le = joblib.load('models/label_encoder.joblib')
xgb_multi = joblib.load('models/xgb_multi.joblib')

# Get feature columns from results
feature_cols = results['feature_columns']

# Prepare test data - handle missing columns
available_cols = [col for col in feature_cols if col in test_df.columns]
X_test = test_df[available_cols].fillna(0).values

# We need to match dimensions, so let's use the original approach
from sklearn.preprocessing import LabelEncoder

# Encode categorical columns in test set
test_data = test_df.copy()
categorical_cols = ['proto', 'service', 'state']
for col in categorical_cols:
    if col in test_data.columns:
        le_temp = LabelEncoder()
        test_data[col + '_encoded'] = le_temp.fit_transform(test_data[col].astype(str))

# Now get feature columns that exist
exclude = ['id', 'label', 'attack_cat', 'proto', 'service', 'state']
feature_cols_actual = [c for c in test_data.columns if c not in exclude]

X_test = test_data[feature_cols_actual].fillna(0).values
y_test = le.transform(test_data['attack_cat'].str.strip())

# Make predictions
y_pred = xgb_multi.predict(X_test)

# ============================================================================
# 7. PER-ATTACK PERFORMANCE
# ============================================================================
print("\n[7] Creating per-attack performance chart...")
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

fig, ax = plt.subplots(figsize=(14, 8), facecolor=COLORS['bg'])
ax.set_facecolor(COLORS['bg'])

attack_names = le.classes_
f1_scores = [report[name]['f1-score'] for name in attack_names]
recalls = [report[name]['recall'] for name in attack_names]
precisions = [report[name]['precision'] for name in attack_names]

x = np.arange(len(attack_names))
width = 0.25

bars1 = ax.bar(x - width, precisions, width, label='Precision', color=COLORS['accent'])
bars2 = ax.bar(x, recalls, width, label='Recall', color=COLORS['green'])
bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color=COLORS['purple'])

ax.set_ylabel('Score', fontsize=12, color=COLORS['text'])
ax.set_title('Per-Attack Detection Performance', fontsize=18, fontweight='bold',
             color=COLORS['text'], pad=20)
ax.set_xticks(x)
ax.set_xticklabels(attack_names, rotation=45, ha='right', fontsize=10)
ax.tick_params(colors=COLORS['text'])
ax.legend(loc='upper right', fontsize=11)
ax.axhline(y=0.8, color=COLORS['yellow'], linestyle='--', alpha=0.5)
ax.set_ylim(0, 1.1)

plt.tight_layout()
save_fig(fig, '07_per_attack_performance')

# ============================================================================
# 8. NETWORK PROTOCOL ANALYSIS
# ============================================================================
print("[8] Creating network protocol analysis...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor=COLORS['bg'])

# Protocol distribution
ax1.set_facecolor(COLORS['bg'])
proto_counts = df['proto'].value_counts().head(10)
colors_proto = plt.cm.viridis(np.linspace(0.2, 0.9, len(proto_counts)))
ax1.pie(proto_counts.values, labels=proto_counts.index, autopct='%1.1f%%',
        colors=colors_proto, textprops={'color': COLORS['text']})
ax1.set_title('Top Network Protocols', fontsize=16, fontweight='bold',
              color=COLORS['text'], pad=15)

# Service distribution
ax2.set_facecolor(COLORS['bg'])
service_counts = df['service'].value_counts().head(10)
bars = ax2.barh(range(len(service_counts)), service_counts.values[::-1],
                color=plt.cm.plasma(np.linspace(0.2, 0.8, len(service_counts))))
ax2.set_yticks(range(len(service_counts)))
ax2.set_yticklabels(service_counts.index[::-1], fontsize=10)
ax2.set_xlabel('Count', fontsize=12, color=COLORS['text'])
ax2.set_title('Top Network Services', fontsize=16, fontweight='bold',
              color=COLORS['text'], pad=15)
ax2.tick_params(colors=COLORS['text'])

plt.tight_layout()
save_fig(fig, '08_network_analysis')

# ============================================================================
# 9. TRAFFIC VOLUME ANALYSIS
# ============================================================================
print("[9] Creating traffic volume analysis...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor=COLORS['bg'])

# Bytes transferred by attack type
ax1.set_facecolor(COLORS['bg'])
bytes_by_attack = df.groupby('attack_cat')[['sbytes', 'dbytes']].mean()
bytes_by_attack = bytes_by_attack.sort_values('sbytes', ascending=True)

y = np.arange(len(bytes_by_attack))
height = 0.35

bars1 = ax1.barh(y - height/2, bytes_by_attack['sbytes'], height, 
                  label='Source Bytes', color=COLORS['accent'])
bars2 = ax1.barh(y + height/2, bytes_by_attack['dbytes'], height,
                  label='Dest Bytes', color=COLORS['green'])

ax1.set_yticks(y)
ax1.set_yticklabels(bytes_by_attack.index, fontsize=10)
ax1.set_xlabel('Average Bytes', fontsize=12, color=COLORS['text'])
ax1.set_title('Average Traffic by Attack Type', fontsize=16, fontweight='bold',
              color=COLORS['text'], pad=15)
ax1.tick_params(colors=COLORS['text'])
ax1.legend()

# Duration by attack type
ax2.set_facecolor(COLORS['bg'])
if 'dur' in df.columns:
    dur_by_attack = df.groupby('attack_cat')['dur'].mean().sort_values()
    colors_dur = [ATTACK_COLORS.get(cat, COLORS['accent']) for cat in dur_by_attack.index]
    bars = ax2.barh(dur_by_attack.index, dur_by_attack.values, color=colors_dur)
    ax2.set_xlabel('Average Duration (seconds)', fontsize=12, color=COLORS['text'])
    ax2.set_title('Average Connection Duration by Attack', fontsize=16, fontweight='bold',
                  color=COLORS['text'], pad=15)
    ax2.tick_params(colors=COLORS['text'])

plt.tight_layout()
save_fig(fig, '09_traffic_volume')

# ============================================================================
# 10. SUMMARY DASHBOARD
# ============================================================================
print("[10] Creating summary dashboard...")

importance_df = pd.DataFrame(results['feature_importance'])

fig = plt.figure(figsize=(20, 14), facecolor=COLORS['bg'])

# Title
fig.text(0.5, 0.97, 'UNSW-NB15 NETWORK INTRUSION DETECTION', fontsize=32, 
         fontweight='bold', ha='center', color=COLORS['text'])
fig.text(0.5, 0.94, 'AI-Powered Cybersecurity - XGBoost Classification', fontsize=16,
         ha='center', color=COLORS['accent'])

# Key metrics boxes
metrics_data = [
    ('90.0%', 'Binary Accuracy', COLORS['green']),
    ('76.4%', 'Multi-Class Accuracy', COLORS['accent']),
    ('98.5%', 'ROC-AUC Score', COLORS['purple']),
    ('9', 'Attack Types Detected', COLORS['yellow']),
    ('257K', 'Records Analyzed', COLORS['orange']),
    ('43', 'Network Features', COLORS['cyan']),
]

for i, (value, label, color) in enumerate(metrics_data):
    ax = fig.add_axes([0.05 + i*0.155, 0.78, 0.14, 0.12])
    ax.set_facecolor(COLORS['card'])
    ax.axis('off')
    ax.text(0.5, 0.65, value, fontsize=28, fontweight='bold', 
            ha='center', va='center', color=color, transform=ax.transAxes)
    ax.text(0.5, 0.25, label, fontsize=10, ha='center', va='center',
            color=COLORS['text'], transform=ax.transAxes)
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(2)

# Attack distribution mini chart
ax1 = fig.add_axes([0.05, 0.35, 0.28, 0.38])
ax1.set_facecolor(COLORS['bg'])
attack_counts = df['attack_cat'].value_counts()
colors_pie = [ATTACK_COLORS.get(cat, COLORS['accent']) for cat in attack_counts.index]
ax1.pie(attack_counts.values, labels=attack_counts.index, colors=colors_pie,
        autopct='%1.1f%%', textprops={'color': COLORS['text'], 'fontsize': 9})
ax1.set_title('Attack Distribution', fontsize=14, fontweight='bold',
              color=COLORS['text'], pad=10)

# Model comparison mini chart
ax2 = fig.add_axes([0.38, 0.35, 0.28, 0.38])
ax2.set_facecolor(COLORS['bg'])
models = ['XGBoost', 'Random Forest', 'LightGBM']
binary_acc = [90.04, 89.93, 89.93]
multi_acc = [76.35, 75.40, 71.70]
x = np.arange(len(models))
width = 0.35
ax2.bar(x - width/2, binary_acc, width, label='Binary', color=COLORS['green'])
ax2.bar(x + width/2, multi_acc, width, label='Multi-Class', color=COLORS['accent'])
ax2.set_ylabel('Accuracy %', color=COLORS['text'])
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=9)
ax2.tick_params(colors=COLORS['text'])
ax2.legend(loc='lower right', fontsize=9)
ax2.set_title('Model Comparison', fontsize=14, fontweight='bold',
              color=COLORS['text'], pad=10)
ax2.set_ylim(60, 100)

# Top features mini chart
ax3 = fig.add_axes([0.71, 0.35, 0.26, 0.38])
ax3.set_facecolor(COLORS['bg'])
top5 = importance_df.nlargest(5, 'importance')
ax3.barh(top5['feature'], top5['importance'], color=COLORS['purple'])
ax3.set_xlabel('Importance', color=COLORS['text'], fontsize=9)
ax3.tick_params(colors=COLORS['text'], labelsize=9)
ax3.set_title('Top 5 Features', fontsize=14, fontweight='bold',
              color=COLORS['text'], pad=10)

# Insights text
insights = [
    "KEY FINDINGS:",
    "  XGBoost achieves 90% accuracy in detecting network intrusions",
    "  'Generic' attacks are most common (38%), followed by 'Exploits' (17%)",
    "  Top predictive features: sttl, ct_state_ttl, sbytes, sload",
    "  Model detects 9 different attack types with varying precision",
    "  Normal traffic correctly identified 98% of the time",
]

ax_text = fig.add_axes([0.05, 0.05, 0.9, 0.25])
ax_text.set_facecolor(COLORS['card'])
ax_text.axis('off')
for i, text in enumerate(insights):
    weight = 'bold' if i == 0 else 'normal'
    color = COLORS['accent'] if i == 0 else COLORS['text']
    ax_text.text(0.02, 0.85 - i*0.14, text, fontsize=12, fontweight=weight,
                 color=color, transform=ax_text.transAxes)

save_fig(fig, '10_summary_dashboard')

print("\n" + "=" * 50)
print("All remaining graphs generated!")
print("=" * 50)

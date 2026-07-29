"""
Standalone offline EDA script (optional / reference only).

The live application performs EDA interactively via `GET /api/analytics`
(see backend/app/core/analytics.py) and renders it with Recharts in the
frontend's /analytics page. This script is kept for anyone who wants to
regenerate static matplotlib/seaborn plots for a paper, slide deck, or
this README, without needing the web app running.

Usage:
    cd backend/notebooks
    python eda_offline.py
Outputs PNGs to ./eda_plots/
"""
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set style for professional plots
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Paths
DATA_PATH = "../data/Thyroid_Diff.csv"
OUT_DIR = "./eda_plots"
os.makedirs(OUT_DIR, exist_ok=True)
print(f" Output directory: {OUT_DIR}")

# Load data
df = pd.read_csv(DATA_PATH)
TARGET = "Recurred"
print(f"\n Loaded {df.shape[0]} rows, {df.shape[1]} columns")
print(f" Columns: {df.columns.tolist()}")
print(f"\n Target distribution:\n{df[TARGET].value_counts()}")

# Identify feature types
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != TARGET]
categorical_cols = [c for c in df.columns if c not in numeric_cols + [TARGET] and df[c].dtype == 'object']

print(f"\n Feature types:")
print(f"  - Numeric: {len(numeric_cols)} ({', '.join(numeric_cols)})")
print(f"  - Categorical: {len(categorical_cols)}")

# ==========================================
# 1. Target Distribution
# ==========================================
print("\n Generating: Target Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Count plot
ax1 = axes[0]
counts = df[TARGET].value_counts()
colors = ['#0f766e', '#dc2626']  # Teal for No, Red for Yes
bars = ax1.bar(counts.index, counts.values, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_title('Target Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Recurrence')
ax1.set_ylabel('Count')
ax1.grid(axis='y', alpha=0.3)

# Add percentage labels
total = len(df)
for bar, count in zip(bars, counts.values):
    height = bar.get_height()
    percentage = (count/total)*100
    ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{count}\n({percentage:.1f}%)',
             ha='center', va='bottom', fontweight='bold')

# Pie chart
ax2 = axes[1]
explode = (0.05, 0.1)
wedges, texts, autotexts = ax2.pie(counts.values, 
                                   labels=counts.index,
                                   autopct='%1.1f%%',
                                   colors=colors,
                                   explode=explode,
                                   shadow=True,
                                   startangle=90,
                                   textprops={'fontsize': 12, 'weight': 'bold'})
ax2.set_title('Class Proportions', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/1_target_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {OUT_DIR}/1_target_distribution.png")

# ==========================================
# 2. Age Distribution Analysis
# ==========================================
print(" Generating: Age Distribution Analysis...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Histogram with KDE
ax1 = axes[0]
df['Age'].hist(bins=30, edgecolor='black', alpha=0.7, color='#3498db', ax=ax1)
ax1.axvline(df['Age'].mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {df["Age"].mean():.1f}')
ax1.axvline(df['Age'].median(), color='green', linestyle='--', linewidth=2, 
           label=f'Median: {df["Age"].median():.1f}')
ax1.set_title('Age Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Age')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(alpha=0.3)

# Box plot by recurrence
ax2 = axes[1]
sns.boxplot(x=TARGET, y='Age', data=df, palette=['#0f766e', '#dc2626'], ax=ax2)
ax2.set_title('Age Distribution by Recurrence', fontsize=14, fontweight='bold')
ax2.set_xlabel('Recurrence')
ax2.set_ylabel('Age')
ax2.grid(alpha=0.3)

# Violin plot with swarm
ax3 = axes[2]
sns.violinplot(x=TARGET, y='Age', data=df, palette=['#0f766e', '#dc2626'], 
               split=True, inner='quartile', ax=ax3)
sns.swarmplot(x=TARGET, y='Age', data=df, color='black', alpha=0.3, size=3, ax=ax3)
ax3.set_title('Age Violin Plot by Recurrence', fontsize=14, fontweight='bold')
ax3.set_xlabel('Recurrence')
ax3.set_ylabel('Age')
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/2_age_analysis.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {OUT_DIR}/2_age_analysis.png")

# ==========================================
# 3. Categorical Features Analysis
# ==========================================
print(" Generating: Categorical Features Analysis...")
if categorical_cols:
    n_cols = 3
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]

    for idx, col in enumerate(categorical_cols):
        if idx < len(axes):
            ax = axes[idx]
            crosstab = pd.crosstab(df[col], df[TARGET], normalize='index') * 100
            crosstab.plot(kind='bar', ax=ax, color=['#0f766e', '#dc2626'], 
                         edgecolor='black', linewidth=0.5)
            ax.set_title(f'{col}\n(Recurrence Rate by Category)', fontsize=11, fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Percentage (%)')
            ax.legend(['No Recurrence', 'Recurrence'], loc='upper right', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Hide unused subplots
    for idx in range(len(categorical_cols), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/3_categorical_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {OUT_DIR}/3_categorical_analysis.png")

# ==========================================
# 4. Correlation Analysis
# ==========================================
print(" Generating: Correlation Analysis...")

# Encode categorical variables for correlation
enc_df = df.copy()
for c in enc_df.columns:
    if not pd.api.types.is_numeric_dtype(enc_df[c]):
        enc_df[c] = LabelEncoder().fit_transform(enc_df[c].astype(str))

# Full correlation heatmap
plt.figure(figsize=(14, 12))
correlation_matrix = enc_df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            fmt=".2f", 
            cmap="coolwarm", 
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 7})
plt.title('Correlation Heatmap (All Features)', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/4_correlation_heatmap_full.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {OUT_DIR}/4_correlation_heatmap_full.png")

# Correlation with target
corr_with_target = correlation_matrix[TARGET].sort_values(ascending=False)
plt.figure(figsize=(12, 6))
colors_corr = ['green' if x > 0 else 'red' for x in corr_with_target.drop(TARGET)]
corr_with_target.drop(TARGET).plot(kind='bar', color=colors_corr, edgecolor='black')
plt.title('Feature Correlation with Target (Recurrence)', fontsize=14, fontweight='bold')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45, ha='right')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.axhline(y=0.2, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
plt.axhline(y=-0.2, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/5_correlation_with_target.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {OUT_DIR}/5_correlation_with_target.png")

# ==========================================
# 5. Feature Distribution by Target (Numeric Only)
# ==========================================
print(" Generating: Feature Distribution by Target...")
if numeric_cols:
    # Get top correlated numeric features
    top_numeric = [f for f in numeric_cols if f in corr_with_target.index]
    top_numeric = sorted(top_numeric, key=lambda x: abs(corr_with_target[x]), reverse=True)[:6]
    
    if top_numeric:
        n_cols = 3
        n_rows = (len(top_numeric) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for idx, feature in enumerate(top_numeric):
            if idx < len(axes):
                ax = axes[idx]
                data_no = df[df[TARGET] == 'No'][feature].dropna()
                data_yes = df[df[TARGET] == 'Yes'][feature].dropna()
                
                bp = ax.boxplot([data_no, data_yes], 
                               labels=['No', 'Yes'],
                               patch_artist=True,
                               medianprops=dict(color='black', linewidth=2))
                bp['boxes'][0].set_facecolor('#0f766e')
                bp['boxes'][1].set_facecolor('#dc2626')
                
                ax.set_title(f'{feature}\n(by Recurrence)', fontsize=11, fontweight='bold')
                ax.set_xlabel('Recurrence')
                ax.set_ylabel('Value')
                ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(top_numeric), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/6_top_features_by_target.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {OUT_DIR}/6_top_features_by_target.png")

# ==========================================
# 6. Dimensionality Reduction (PCA & t-SNE)
# ==========================================
print(" Generating: Dimensionality Reduction...")
X_encoded = enc_df.drop(columns=[TARGET])
y_encoded = enc_df[TARGET]

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_encoded)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PCA plot
ax1 = axes[0]
scatter1 = ax1.scatter(pca_result[:, 0], pca_result[:, 1], 
                      c=y_encoded, cmap='RdYlGn', alpha=0.7, 
                      edgecolors='black', linewidth=0.5, s=50)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
ax1.set_title('PCA Visualization', fontsize=14, fontweight='bold')
ax1.grid(alpha=0.3)
plt.colorbar(scatter1, ax=ax1, label='Recurrence (0=No, 1=Yes)')

# t-SNE
try:
    if len(df) > 500:
        sample_idx = np.random.choice(len(df), 500, replace=False)
        tsne_data = X_encoded.iloc[sample_idx]
        tsne_labels = y_encoded.iloc[sample_idx]
    else:
        tsne_data = X_encoded
        tsne_labels = y_encoded
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_result = tsne.fit_transform(tsne_data)
    
    ax2 = axes[1]
    scatter2 = ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                          c=tsne_labels, cmap='RdYlGn', alpha=0.7, 
                          edgecolors='black', linewidth=0.5, s=50)
    ax2.set_xlabel('t-SNE Component 1')
    ax2.set_ylabel('t-SNE Component 2')
    ax2.set_title('t-SNE Visualization', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Recurrence (0=No, 1=Yes)')
except Exception as e:
    print(f"   t-SNE error: {e}")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/7_dimensionality_reduction.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {OUT_DIR}/7_dimensionality_reduction.png")

# ==========================================
# 7. Pair Plot (Top Numeric Features)
# ==========================================
print(" Generating: Pair Plot...")
if len(numeric_cols) >= 2:
    top_pair = [f for f in numeric_cols if f in corr_with_target.index]
    top_pair = sorted(top_pair, key=lambda x: abs(corr_with_target[x]), reverse=True)[:4]
    
    if len(top_pair) >= 2:
        pairplot_data = df[top_pair + [TARGET]].copy()
        g = sns.pairplot(pairplot_data, 
                        hue=TARGET, 
                        palette=['#0f766e', '#dc2626'],
                        diag_kind='kde',
                        markers=['o', 's'],
                        plot_kws={'alpha': 0.6, 'edgecolor': 'black', 'linewidth': 0.5})
        g.fig.suptitle('Pair Plot of Top Features', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/8_pairplot_top_features.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {OUT_DIR}/8_pairplot_top_features.png")

# ==========================================
# 8. Missing Value Analysis
# ==========================================
print(" Generating: Missing Value Analysis...")
missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0]

if len(missing_data) > 0:
    plt.figure(figsize=(10, 6))
    missing_data.plot(kind='bar', color='orange', edgecolor='black')
    plt.title('Missing Values by Feature', fontsize=14, fontweight='bold')
    plt.xlabel('Features')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=45, ha='right')
    for i, v in enumerate(missing_data):
        plt.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/9_missing_values.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {OUT_DIR}/9_missing_values.png")
else:
    print("   No missing values found")

# ==========================================
# 9. Class Imbalance Analysis
# ==========================================
print(" Generating: Class Imbalance Analysis...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Class distribution bar
ax = axes[0]
colors_bar = ['#0f766e' if label == 'No' else '#dc2626' for label in counts.index]
bars = ax.bar(counts.index, counts.values, color=colors_bar, edgecolor='black', linewidth=1.5)
ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Recurrence Class')
ax.set_ylabel('Count')
ax.grid(axis='y', alpha=0.3)
for bar, count in zip(bars, counts.values):
    height = bar.get_height()
    percentage = (count/total)*100
    ax.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'{count}\n({percentage:.1f}%)', ha='center', fontweight='bold')

# 2. Imbalance ratio
ax = axes[1]
no_count = counts.get('No', 0)
yes_count = counts.get('Yes', 0)
ratio = no_count / yes_count if yes_count > 0 else float('inf')

imbalance_text = f'No/Yes Ratio: {ratio:.2f}\n\n'
if ratio > 5:
    imbalance_text += ' Severe Imbalance\n(Needs SMOTE/Random Sampling)'
elif ratio > 3:
    imbalance_text += ' High Imbalance\n(Consider Class Weights)'
elif ratio > 1.5:
    imbalance_text += ' Moderate Imbalance\n(Use Balanced Class Weight)'
else:
    imbalance_text += ' Balanced Dataset\n(No special handling needed)'

ax.text(0.5, 0.5, imbalance_text, horizontalalignment='center', verticalalignment='center',
        transform=ax.transAxes, fontsize=12, fontweight='bold')
ax.axis('off')
ax.set_title('Imbalance Analysis', fontsize=14, fontweight='bold')

# 3. Pie chart
ax = axes[2]
wedges, texts, autotexts = ax.pie(counts.values, 
                                 labels=counts.index,
                                 autopct='%1.1f%%',
                                 colors=['#0f766e', '#dc2626'],
                                 explode=(0, 0.1),
                                 shadow=True,
                                 textprops={'fontsize': 12, 'weight': 'bold'})
ax.set_title('Class Proportions', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/10_class_imbalance_analysis.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {OUT_DIR}/10_class_imbalance_analysis.png")

# ==========================================
# Summary Report
# ==========================================
print("\n" + "=" * 70)
print(" EDA SUMMARY REPORT")
print("=" * 70)
print(f"""
Dataset Overview:
- Total Samples: {df.shape[0]}
- Total Features: {df.shape[1]}
- Feature Types: 
  * Numeric: {len(numeric_cols)} ({', '.join(numeric_cols)})
  * Categorical: {len(categorical_cols)}

Target Distribution:
- No Recurrence: {no_count} ({no_count/len(df)*100:.1f}%)
- Recurrence: {yes_count} ({yes_count/len(df)*100:.1f}%)
- Imbalance Ratio: {ratio:.2f}

Age Statistics:
- Mean: {df['Age'].mean():.2f}
- Median: {df['Age'].median():.2f}
- Std Dev: {df['Age'].std():.2f}
- Range: {df['Age'].min():.0f} - {df['Age'].max():.0f}

Top 5 Correlated Features:
{corr_with_target.drop(TARGET).head(5).to_string()}

 All plots saved to: {OUT_DIR}
""")

print("=" * 70)
print(" EDA Complete!")
print("=" * 70)

# List all generated plots
print("\n Generated plots:")
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith('.png'):
        size = os.path.getsize(f"{OUT_DIR}/{f}") / 1024  # KB
        print(f"   {f} ({size:.1f} KB)")
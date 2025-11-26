import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
OUTPUT_DIR = 'output'
PLOT_DIR = 'presentation_plots'
os.makedirs(PLOT_DIR, exist_ok=True)

# Set global style
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 300

def get_accuracy(cm):
    """Calculate accuracy from confusion matrix list."""
    cm = np.array(cm)
    if cm.sum() == 0: return 0.0
    return np.trace(cm) / np.sum(cm)

def load_model_data(filename):
    """Loads trials from a JSON file into a Pandas DataFrame."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Warning: {filename} not found. Skipping.")
        return None, None

    with open(filepath, 'r') as f:
        data = json.load(f)
    
    model_name = data['model_name']
    trials = []
    for t in data.get('trials', []):
        row = t['hyperparameters'].copy()
        row['accuracy'] = get_accuracy(t['confusion_matrix'])
        trials.append(row)
    
    return model_name, pd.DataFrame(trials)

# ==========================================
# 1. Logistic Regression (Slide 7)
# ==========================================
def plot_logistic_regression():
    name, df = load_model_data('logistic_regression.json')
    if df is None: return

    plt.figure()
    ax = sns.barplot(data=df, x='C', y='accuracy', hue='solver', palette='viridis')
    plt.title(f"{name}: Accuracy vs. Regularization (C)", fontsize=16, pad=20)
    plt.ylabel("Accuracy")
    plt.xlabel("Inverse Regularization Strength (C)")
    plt.legend(title='Solver', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=3, fontsize=12)

    y_min, y_max = df['accuracy'].min(), df['accuracy'].max()
    buffer = (y_max - y_min) * 5 if y_max != y_min else 0.01
    plt.ylim(y_min - buffer, y_max + buffer)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/slide7_logistic_stability.png")
    plt.close()

# ==========================================
# 2. KNN (Slide 8)
# ==========================================
def plot_knn():
    name, df = load_model_data('k_nearest_neighbors.json')
    if df is None: return

    plt.figure()
    df['Distance Metric'] = df['p'].map({1: 'Manhattan (p=1)', 2: 'Euclidean (p=2)'})
    sns.lineplot(data=df, x='n_neighbors', y='accuracy', hue='Distance Metric', 
                 style='Distance Metric', markers=True, dashes=False, palette='viridis', linewidth=3, markersize=10)
    plt.title(f"{name}: Impact of Distance Metric & Neighbors", fontsize=16, pad=20)
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Neighbors (k)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/slide8_knn_distance.png")
    plt.close()

# ==========================================
# 3. Naive Bayes (Slide 9)
# ==========================================
def plot_naive_bayes():
    name, df = load_model_data('gaussian_naive_bayes.json')
    if df is None: return

    plt.figure()
    sns.lineplot(data=df, x='var_smoothing', y='accuracy', marker='o', markersize=10, linewidth=3, color='crimson')
    plt.xscale('log')
    plt.title(f"{name}: Sensitivity to Variance Smoothing", fontsize=16, pad=20)
    plt.ylabel("Accuracy")
    plt.xlabel("Variance Smoothing (Log Scale)")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/slide9_nb_smoothing.png")
    plt.close()

# ==========================================
# 4. Random Forest (Slide 10)
# ==========================================
def plot_random_forest():
    name, df = load_model_data('random_forest.json')
    if df is None: return

    plt.figure()
    df['depth_str'] = df['max_depth'].fillna('Unlimited').astype(str)
    df['Config'] = df.apply(lambda x: f"{x['n_estimators']} Trees\n(Depth {x['depth_str']})", axis=1)
    df = df.sort_values('accuracy')

    ax = sns.barplot(data=df, x='Config', y='accuracy', palette='Greens_d')
    
    plt.title(f"{name}: Impact of Model Complexity", fontsize=16, pad=20)
    plt.ylabel("Accuracy")
    plt.xlabel("Configuration (Trees & Depth)")
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=3, fontsize=12)

    plt.ylim(0, 0.6) 
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/slide10_rf_complexity.png")
    plt.close()

# ==========================================
# 5. XGBoost (Slide 11) - TWO PLOTS
# ==========================================
def plot_xgboost():
    name, df = load_model_data('xgboost.json')
    if df is None: return

    # --- Plot 1: Learning Rate ---
    plt.figure()
    sns.swarmplot(data=df, x='learning_rate', y='accuracy', hue='max_depth', palette='deep', size=9)
    sns.boxplot(data=df, x='learning_rate', y='accuracy', showfliers=False, color='lightgray', boxprops={'alpha': 0.3})
    plt.title(f"{name}: Performance by Learning Rate", fontsize=16, pad=20)
    plt.ylabel("Accuracy")
    plt.xlabel("Learning Rate")
    plt.legend(title='Max Depth')
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/slide11_xgb_learning_rate.png")
    print(f"Generated: slide11_xgb_learning_rate.png")
    plt.close()

    # --- Plot 2: Iterations (n_estimators) ---
    plt.figure()
    # Boxplot to show distribution per estimator count
    sns.boxplot(data=df, x='n_estimators', y='accuracy', showfliers=False, color='lightgray', boxprops={'alpha': 0.3})
    # Swarmplot to show individual trials colored by depth
    sns.swarmplot(data=df, x='n_estimators', y='accuracy', hue='max_depth', palette='magma', size=9)
    
    plt.title(f"{name}: Impact of Boosting Rounds (Iterations)", fontsize=16, pad=20)
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Estimators (Trees)")
    plt.legend(title='Max Depth', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/slide11_xgb_iterations.png")
    print(f"Generated: slide11_xgb_iterations.png")
    plt.close()

# ==========================================
# 6. MLP (Slide 12)
# ==========================================
def plot_mlp():
    name, df = load_model_data('multilayer_perceptron.json')
    if df is None: return

    y_min = df['accuracy'].min() - 0.01
    y_max = df['accuracy'].max() + 0.01

    plt.figure()
    sns.barplot(data=df, x='hidden_dim', y='accuracy', palette='Blues_d', errorbar='sd')
    plt.title(f"{name}: Impact of Network Width", fontsize=16, pad=20)
    plt.xlabel("Hidden Dimension (Neurons)")
    plt.ylabel("Accuracy")
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/slide12_mlp_width.png")
    print(f"Generated: slide12_mlp_width.png")
    plt.close()

    plt.figure()
    sns.barplot(data=df, x='num_layers', y='accuracy', palette='Reds_d', errorbar='sd')
    plt.title(f"{name}: Impact of Network Depth", fontsize=16, pad=20)
    plt.xlabel("Number of Layers")
    plt.ylabel("Accuracy")
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/slide12_mlp_depth.png")
    print(f"Generated: slide12_mlp_depth.png")
    plt.close()

def main():
    print(">>> Generating Presentation Plots...")
    plot_logistic_regression()
    plot_knn()
    plot_naive_bayes()
    plot_random_forest()
    plot_xgboost()
    plot_mlp()
    print(f"\n>>> Done. Plots saved to '{PLOT_DIR}/'")

if __name__ == "__main__":
    main()

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
OUTPUT_DIR = 'output'
PLOT_DIR = 'presentation_plots'
MODEL_FILE = 'xgboost.json'
METADATA_FILE = 'data/metadata.json'

os.makedirs(PLOT_DIR, exist_ok=True)

# Set style
sns.set_theme(style="white", context="talk")
plt.rcParams['figure.figsize'] = (14, 12) # Made slightly larger for readability
plt.rcParams['figure.dpi'] = 300

def get_accuracy(cm):
    cm = np.array(cm)
    if cm.sum() == 0: return 0.0
    return np.trace(cm) / np.sum(cm)

def load_genre_labels():
    if not os.path.exists(METADATA_FILE):
        print(f"Error: {METADATA_FILE} not found. Using generic labels.")
        return None
    
    with open(METADATA_FILE, 'r') as f:
        meta = json.load(f)
    
    labels_dict = meta.get('genre_labels', {})
    sorted_labels = [labels_dict[str(i)] for i in range(len(labels_dict))]
    return sorted_labels

def plot_best_cm():
    filepath = os.path.join(OUTPUT_DIR, MODEL_FILE)
    if not os.path.exists(filepath):
        print(f"Error: {MODEL_FILE} not found.")
        return

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Find best trial
    best_trial = None
    best_acc = -1.0
    
    for trial in data.get('trials', []):
        acc = get_accuracy(trial['confusion_matrix'])
        if acc > best_acc:
            best_acc = acc
            best_trial = trial

    if best_trial is None:
        print("No valid trials found.")
        return

    print(f"Best Accuracy: {best_acc:.4f}")

    # Raw Counts
    cm = np.array(best_trial['confusion_matrix'])
    
    # Load Labels
    labels = load_genre_labels()
    if labels is None:
        labels = [f"Class {i}" for i in range(cm.shape[0])]

    plt.figure()
    
    # Plot RAW COUNTS
    sns.heatmap(
        cm, 
        annot=False, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=labels, 
        yticklabels=labels,
        cbar_kws={'label': 'Count'} # Corrected Label
    )
    
    plt.title(f"XGBoost Confusion Matrix (Accuracy: {best_acc:.2%})", fontsize=18, pad=20)
    plt.xlabel("Predicted Genre", fontsize=14)
    plt.ylabel("Actual Genre", fontsize=14)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    
    save_path = os.path.join(PLOT_DIR, 'best_xgboost_cm.png')
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_best_cm()

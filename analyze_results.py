import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
OUTPUT_DIR = 'output'
METADATA_FILE = 'data/metadata.json'
RESULTS_IMG_DIR = 'analysis_images'

# Ensure image directory exists
os.makedirs(RESULTS_IMG_DIR, exist_ok=True)

def calculate_metrics_from_cm(cm_list):
    """
    Calculates Accuracy, Weighted F1, and Macro F1 from a raw confusion matrix list.
    """
    cm = np.array(cm_list)
    
    # True Positives (Diagonal)
    tp = np.diag(cm)
    
    # False Positives (Column sum - TP)
    fp = np.sum(cm, axis=0) - tp
    
    # False Negatives (Row sum - TP)
    fn = np.sum(cm, axis=1) - tp
    
    # Support (Actual count per class)
    support = np.sum(cm, axis=1)
    total_samples = np.sum(support)
    
    # Accuracy
    accuracy = np.sum(tp) / total_samples
    
    # Precision & Recall (handle division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where((tp + fp) > 0, tp / (tp + fp), 0)
        recall = np.where((tp + fn) > 0, tp / (tp + fn), 0)
        f1_per_class = np.where((precision + recall) > 0, 
                                2 * (precision * recall) / (precision + recall), 
                                0)
    
    # Macro F1 (Unweighted average of F1 scores)
    macro_f1 = np.mean(f1_per_class)
    
    # Weighted F1 (Average F1 score weighted by class support)
    weighted_f1 = np.sum(f1_per_class * support) / total_samples
    
    return accuracy, weighted_f1, macro_f1

def load_results():
    """Loads all JSON files from the output directory."""
    results = []
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: Directory '{OUTPUT_DIR}' not found.")
        return pd.DataFrame()

    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')]
    
    print(f"Found {len(files)} model result files.")
    
    for file in files:
        filepath = os.path.join(OUTPUT_DIR, file)
        with open(filepath, 'r') as f:
            data = json.load(f)
            
            # Calculate metrics from the best confusion matrix
            acc, w_f1, m_f1 = calculate_metrics_from_cm(data['best_confusion_matrix'])
            
            results.append({
                'Model': data['model_name'],
                'Author': data['person_name'],
                'Accuracy': acc,
                'Weighted F1': w_f1,
                'Macro F1': m_f1,
                'Train Time (s)': data['total_train_time'],
                'Test Time (s)': data['total_test_time'],
                'Hyperparameters': data['best_hyperparameters']
            })
            
    return pd.DataFrame(results)

def plot_performance(df):
    """Generates comparison plots."""
    sns.set_theme(style="whitegrid")
    
    # 1. Accuracy vs F1 Score Bar Chart
    plt.figure(figsize=(12, 6))
    
    # Melt dataframe for seaborn
    df_melted = df.melt(id_vars=['Model'], value_vars=['Accuracy', 'Weighted F1'], 
                        var_name='Metric', value_name='Score')
    
    ax = sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric', palette='viridis')
    plt.title('Model Performance Comparison: Accuracy vs Weighted F1', fontsize=16)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15)
    plt.legend(loc='lower right')
    
    # Add labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
        
    plt.tight_layout()
    plt.savefig(f'{RESULTS_IMG_DIR}/model_comparison_scores.png')
    print(f"Saved score comparison plot to {RESULTS_IMG_DIR}/model_comparison_scores.png")
    plt.show()

    # 2. Efficiency Frontier (Accuracy vs Prediction Time)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Test Time (s)', y='Accuracy', hue='Model', style='Model', s=200, palette='deep')
    
    # Add labels
    for i in range(df.shape[0]):
        plt.text(
            df['Test Time (s)'][i], 
            df['Accuracy'][i]+0.005, 
            df['Model'][i], 
            horizontalalignment='left', 
            size='medium', 
            color='black', 
            weight='semibold'
        )

    plt.title('Efficiency Frontier: Accuracy vs. Inference Speed', fontsize=16)
    plt.xlabel('Total Test Time (seconds) [Lower is Faster]')
    plt.ylabel('Accuracy [Higher is Better]')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_IMG_DIR}/model_efficiency.png')
    print(f"Saved efficiency plot to {RESULTS_IMG_DIR}/model_efficiency.png")
    plt.show()

def print_summary(df):
    """Prints a markdown-formatted summary table."""
    # Sort by Accuracy
    df_sorted = df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
    
    print("\n" + "="*80)
    print("FINAL MODEL LEADERBOARD")
    print("="*80)
    
    # Select columns for display
    display_cols = ['Model', 'Author', 'Accuracy', 'Weighted F1', 'Train Time (s)', 'Test Time (s)']
    print(df_sorted[display_cols].to_markdown(index=False, floatfmt=".4f"))
    
    print("\n" + "="*80)
    print("BEST HYPERPARAMETERS BY MODEL")
    print("="*80)
    
    for _, row in df_sorted.iterrows():
        print(f"\n🔹 {row['Model']} (Acc: {row['Accuracy']:.4f})")
        print(json.dumps(row['Hyperparameters'], indent=2))

def main():
    print(">>> STARTING ANALYSIS <<<")
    df = load_results()
    
    if not df.empty:
        print_summary(df)
        plot_performance(df)
        print("\n Analysis Complete.")
    else:
        print("No data found to analyze.")

if __name__ == "__main__":
    main()

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_DIR = 'data'
OUTPUT_DIR = 'output'
IMG_DIR = 'report_images'
os.makedirs(IMG_DIR, exist_ok=True)

# Set global style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def load_data():
    print("Loading CSV data...")
    X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'))
    y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'))
    
    with open(os.path.join(DATA_DIR, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    genre_map = {int(k): v for k, v in meta['genre_labels'].items()}
    
    return X_train, y_train, y_test, genre_map

def load_model_results():
    print("Loading Model JSONs...")
    results = []
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')]
    
    for file in files:
        with open(os.path.join(OUTPUT_DIR, file), 'r') as f:
            data = json.load(f)
            
            # Recalc accuracy from CM to be safe
            cm = np.array(data['best_confusion_matrix'])
            acc = np.trace(cm) / np.sum(cm)
            
            results.append({
                'Model': data['model_name'],
                'Accuracy': acc,
                'Train Time': data['total_train_time'],
                'Test Time': data['total_test_time'],
                'Trials': data.get('trials', [])
            })
    return pd.DataFrame(results)

# --- Plotting Functions ---

def plot_class_balance(y_train, y_test, genre_map):
    print("Generating Figure 1: Class Balance...")
    train_counts = y_train['genre_encoded'].map(genre_map).value_counts().reset_index()
    train_counts.columns = ['Genre', 'Count']
    train_counts['Set'] = 'Train'
    
    test_counts = y_test['genre_encoded'].map(genre_map).value_counts().reset_index()
    test_counts.columns = ['Genre', 'Count']
    test_counts['Set'] = 'Test'
    
    df = pd.concat([train_counts, test_counts])
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Genre', y='Count', hue='Set', palette='viridis')
    plt.xticks(rotation=90)
    plt.title("Class Distribution: Training vs Testing Sets")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig1_class_balance.png'))
    plt.close()

def plot_correlation(X_train):
    print("Generating Figure 2: Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    corr = X_train.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig2_correlation.png'))
    plt.close()

def plot_boxplots(X_train, y_train, genre_map):
    print("Generating Figure 3: Feature Boxplots...")
    df = X_train.copy()
    df['Genre'] = y_train['genre_encoded'].map(genre_map)
    
    # Select 3 distinct genres to make the plot readable
    target_genres = ['Classical', 'Electronic', 'Rock']
    subset = df[df['Genre'].isin(target_genres)]
    
    features = ['energy', 'acousticness', 'loudness']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, feature in enumerate(features):
        sns.boxplot(data=subset, x='Genre', y=feature, ax=axes[i], palette='Set2')
        axes[i].set_title(f'{feature.capitalize()} Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig3_boxplots.png'))
    plt.close()

def plot_model_comparison(results_df):
    print("Generating Figure 4: Model Comparison...")
    df = results_df.sort_values('Accuracy', ascending=False)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x='Accuracy', y='Model', palette='magma')
    plt.xlim(0, 0.6)
    plt.title("Model Accuracy Comparison (Test Set)")
    
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', padding=3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig4_model_comparison.png'))
    plt.close()

def plot_efficiency(results_df):
    print("Generating Figure 5: Efficiency Frontier...")
    plt.figure(figsize=(10, 6))
    
    # Log scale for time because KNN/LogReg are vastly different
    sns.scatterplot(data=results_df, x='Test Time', y='Accuracy', hue='Model', style='Model', s=300, palette='deep')
    
    plt.xscale('log')
    plt.xlabel("Inference Time (Seconds) - Log Scale")
    plt.ylabel("Accuracy")
    plt.title("Efficiency Frontier: Accuracy vs. Prediction Speed")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # Annotate
    for i, row in results_df.iterrows():
        plt.text(row['Test Time'], row['Accuracy']+0.005, row['Model'], fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig5_efficiency.png'))
    plt.close()

def plot_hyperparameters(results_df):
    print("Generating Hyperparameter Plots...")
    
    # 1. XGBoost Learning Rate
    xgb_row = results_df[results_df['Model'] == 'XGBoost'].iloc[0]
    trials = pd.DataFrame([t['hyperparameters'] for t in xgb_row['Trials']])
    trials['acc'] = [np.trace(np.array(t['confusion_matrix']))/np.sum(np.array(t['confusion_matrix'])) for t in xgb_row['Trials']]
    
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=trials, x='learning_rate', y='acc', marker='o')
    plt.title("XGBoost: Learning Rate vs Accuracy")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig6_xgb_lr.png'))
    plt.close()

    # 2. Random Forest Estimators
    rf_row = results_df[results_df['Model'] == 'Random Forest Classifier'].iloc[0]
    trials = pd.DataFrame([t['hyperparameters'] for t in rf_row['Trials']])
    trials['acc'] = [np.trace(np.array(t['confusion_matrix']))/np.sum(np.array(t['confusion_matrix'])) for t in rf_row['Trials']]
    
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=trials, x='n_estimators', y='acc', marker='o', color='green')
    plt.title("Random Forest: N_Estimators vs Accuracy")
    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig7_rf_estimators.png'))
    plt.close()

    # 3. KNN Neighbors
    knn_row = results_df[results_df['Model'] == 'K-Nearest Neighbors'].iloc[0]
    trials = pd.DataFrame([t['hyperparameters'] for t in knn_row['Trials']])
    trials['acc'] = [np.trace(np.array(t['confusion_matrix']))/np.sum(np.array(t['confusion_matrix'])) for t in knn_row['Trials']]
    
    # Filter for just the best metric to see K impact clearly
    best_metric = knn_row['Trials'][-1]['hyperparameters']['metric'] # Assuming last is best or close
    subset = trials[trials['metric'] == best_metric]
    
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=subset, x='n_neighbors', y='acc', marker='o', color='orange')
    plt.title(f"KNN ({best_metric}): K Neighbors vs Accuracy")
    plt.xlabel("K (Neighbors)")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig8_knn_k.png'))
    plt.close()

def main():
    X_train, y_train, y_test, genre_map = load_data()
    results_df = load_model_results()
    
    plot_class_balance(y_train, y_test, genre_map)
    plot_correlation(X_train)
    plot_boxplots(X_train, y_train, genre_map)
    plot_model_comparison(results_df)
    plot_efficiency(results_df)
    plot_hyperparameters(results_df)
    
    print("All figures generated in 'report_images/'")

if __name__ == "__main__":
    main()

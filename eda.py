import pandas as pd
import numpy as np
import json
import os

# Configuration
DATA_DIR = 'data'
METADATA_FILE = os.path.join(DATA_DIR, 'metadata.json')

def load_data():
    print(">>> LOADING DATA...")
    try:
        X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'))
        X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
        y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'))
        
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        genre_map = {int(k): v for k, v in metadata['genre_labels'].items()}
        
        print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
        print(f"X_test:  {X_test.shape}  | y_test:  {y_test.shape}")
        return X_train, y_train, X_test, y_test, genre_map
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

def print_header(title):
    print(f"\n{'='*80}")
    print(f" {title.upper()}")
    print(f"{'='*80}")

def analyze_features(df, name="TRAIN"):
    print_header(f"1. FEATURE STATISTICS ({name})")
    
    stats = df.describe().T
    stats['median'] = df.median()
    stats['skew'] = df.skew()
    stats['kurtosis'] = df.kurt()
    
    # Check if Scaled (Mean approx 0, Std approx 1)
    stats['is_scaled'] = (stats['mean'].abs() < 0.1) & (stats['std'].between(0.9, 1.1))
    
    # Select columns to print
    cols = ['mean', 'std', 'min', 'median', 'max', 'skew', 'is_scaled']
    print(stats[cols].to_markdown(floatfmt=".3f"))

def analyze_correlations(df):
    print_header("2. TOP FEATURE CORRELATIONS")
    
    corr_matrix = df.corr()
    
    # Unstack and remove self-correlations
    corr_pairs = corr_matrix.unstack()
    corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
    
    # Remove duplicates (A-B is same as B-A)
    unique_pairs = corr_pairs.drop_duplicates()
    
    # Sort
    sorted_corr = unique_pairs.sort_values(ascending=False)
    
    print("--- Strongest POSITIVE Correlations ---")
    print(sorted_corr.head(10).to_markdown(floatfmt=".4f"))
    
    print("\n--- Strongest NEGATIVE Correlations ---")
    print(sorted_corr.tail(10).to_markdown(floatfmt=".4f"))

def analyze_class_balance(y_train, y_test, genre_map):
    print_header("3. CLASS BALANCE (TRAIN vs TEST)")
    
    # Map integers to names
    train_counts = y_train['genre_encoded'].map(genre_map).value_counts().sort_index()
    test_counts = y_test['genre_encoded'].map(genre_map).value_counts().sort_index()
    
    df = pd.DataFrame({
        'Train Count': train_counts,
        'Train %': (train_counts / len(y_train)) * 100,
        'Test Count': test_counts,
        'Test %': (test_counts / len(y_test)) * 100
    })
    
    # Check deviation
    df['Diff %'] = df['Train %'] - df['Test %']
    
    print(df.to_markdown(floatfmt=".2f"))
    
    # Imbalance check
    max_c = df['Train Count'].max()
    min_c = df['Train Count'].min()
    ratio = max_c / min_c
    print(f"\nImbalance Ratio (Max/Min): {ratio:.2f}")
    if ratio > 1.5:
        print("(!) Dataset is imbalanced. Ensure models handle class weights or evaluation metrics account for this (e.g., Macro/Weighted F1).")
    else:
        print("(OK) Dataset is relatively balanced.")

def analyze_feature_importance_proxy(X, y, genre_map):
    print_header("4. FEATURE SEPARABILITY (Proxy for Importance)")
    print("Calculating how much each feature varies *between* genres vs *within* genres.")
    
    # Combine X and y
    data = X.copy()
    data['genre'] = y['genre_encoded'].map(genre_map)
    
    # Group by genre and calculate mean for each feature
    means_by_genre = data.groupby('genre').mean()
    
    # Calculate the standard deviation of these means across genres
    # High Std Dev of Means -> Feature values change a lot depending on the genre -> Good feature
    variability = means_by_genre.std().sort_values(ascending=False)
    
    df_var = pd.DataFrame(variability, columns=['Inter-Genre Variability (Std of Means)'])
    print(df_var.to_markdown(floatfmt=".4f"))
    
    print("\nInterpretation:")
    print(f" - Top feature '{df_var.index[0]}' varies significantly between genres (Good predictor).")
    print(f" - Bottom feature '{df_var.index[-1]}' looks similar across all genres (Weak predictor).")

def main():
    X_train, y_train, X_test, y_test, genre_map = load_data()
    
    # 1. Stats
    analyze_features(X_train, "TRAINING SET")
    
    # 2. Correlations
    analyze_correlations(X_train)
    
    # 3. Classes
    analyze_class_balance(y_train, y_test, genre_map)
    
    # 4. Feature Importance Proxy
    analyze_feature_importance_proxy(X_train, y_train, genre_map)
    
    print("\n>>> INSPECTION COMPLETE <<<")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
SAMPLE_SIZE = 2000
RANDOM_STATE = 42
OUTPUT_DIR = 'preprocessed_data'

def load_data(filepath='SpotifyFeatures.csv'):
    """Load the Spotify dataset"""
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Genres: {df['genre'].nunique()} unique categories")
    return df

def clean_data(df):
    """Clean the dataset by removing all multi-genre tracks."""
    print("\n=== Data Cleaning ===")
    
    df['genre'] = df['genre'].replace({"Children’s Music": "Children's Music"})
    print("Combined 'Children’s Music' and 'Children\'s Music' genres.")

    track_id_counts = df['track_id'].value_counts()
    total_unique_songs = len(track_id_counts)
    multi_genre_ids = track_id_counts[track_id_counts > 1].index
    num_multi_genre_songs = len(multi_genre_ids)
    
    if total_unique_songs > 0:
        percentage = (num_multi_genre_songs / total_unique_songs) * 100
        print(f"Found {num_multi_genre_songs} unique songs ({percentage:.2f}%) listed under multiple genres.")
        print("Dropping all entries for these tracks to ensure a clean, single-label dataset.")
        print("-" * 20)
    
    df = df[~df['track_id'].isin(multi_genre_ids)]
    df = df.dropna()
    print(f"Cleaned dataset size: {len(df)} unique, single-genre tracks.")
    
    columns_to_drop = ['track_id', 'track_name', 'artist_name']
    df = df.drop(columns=columns_to_drop)
    
    return df

def engineer_genres(df):
    """Merge similar genres and drop genres with too few samples."""
    print("\n=== Genre Engineering ===")
    
    df['genre'] = df['genre'].replace({'Rap': 'Hip-Hop'})
    print("Merged 'Rap' genre into 'Hip-Hop'.")
    
    genres_to_discard = ['A Capella']
    original_rows = len(df)
    df = df[~df['genre'].isin(genres_to_discard)]
    print(f"Dropped 'A Capella' genre ({original_rows - len(df)} samples removed).")
    
    return df, genres_to_discard

def explore_data(df):
    """Print basic statistics about the dataset after initial cleaning."""
    print("\n=== Dataset Overview (Post-Cleaning & Engineering) ===")
    print(f"Total samples: {len(df)}")
    print(f"\nGenre distribution:")
    genre_counts = df['genre'].value_counts()
    print(genre_counts)
    
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    print("No missing values found" if missing.sum() == 0 else missing[missing > 0])
    
    return genre_counts

def balance_data_hybrid(df, sample_size=2000, random_state=42):
    """
    Balance the dataset using a hybrid strategy:
    - Undersample genres with more than `sample_size` samples.
    - Keep all samples for genres with `sample_size` or fewer samples.
    """
    print("\n=== Balancing Dataset (Hybrid Strategy) ===")
    print(f"Applying hybrid sampling with a threshold of {sample_size} samples per genre...")

    grouped = df.groupby('genre')
    balanced_dfs = []

    for genre, group_df in grouped:
        if len(group_df) > sample_size:
            balanced_dfs.append(group_df.sample(n=sample_size, random_state=random_state))
        else:
            balanced_dfs.append(group_df)
    
    balanced_df = pd.concat(balanced_dfs)
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print("\n--- Balancing Complete ---")
    print(f"Final dataset size: {len(balanced_df)} samples")
    print(f"Final number of genres: {balanced_df['genre'].nunique()}")
    print("\nFinal Genre Distribution:")
    print(balanced_df['genre'].value_counts())
    
    return balanced_df

# --- FIX: Added helper function to parse time_signature ---
def parse_time_signature(time_sig_str):
    """Parse time signature from string format (e.g., '4/4') to numerator"""
    if isinstance(time_sig_str, str):
        return int(time_sig_str.split('/')[0])
    return time_sig_str

def encode_categorical(df):
    """Encode categorical variables like 'key', 'mode', and 'time_signature'."""
    print("\n=== Encoding Categorical Variables ===")
    
    df = df.copy()
    encoders = {}

    # --- FIX: Apply the parsing function to the time_signature column ---
    df['time_signature'] = df['time_signature'].apply(parse_time_signature)
    print("Parsed 'time_signature' from string to integer.")
    
    # Encode 'key'
    key_encoder = LabelEncoder()
    df['key_encoded'] = key_encoder.fit_transform(df['key'])
    encoders['key'] = key_encoder
    
    # Encode 'mode'
    mode_encoder = LabelEncoder()
    df['mode_encoded'] = mode_encoder.fit_transform(df['mode'])
    encoders['mode'] = mode_encoder
    
    df = df.drop(columns=['key', 'mode'])
    print("Encoded 'key' and 'mode' features.")
    
    return df, encoders

def prepare_features(df):
    """Separate features and target, and encode the target variable."""
    print("\n=== Preparing Features and Target ===")
    
    genre_encoder = LabelEncoder()
    y = genre_encoder.fit_transform(df['genre'])
    print(f"Target variable encoded: {len(genre_encoder.classes_)} genres")
    
    X = df.drop(columns=['genre'])
    feature_names = X.columns.tolist()
    print(f"Features: {len(feature_names)} total")
    
    return X.values, y, feature_names, genre_encoder

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into stratified train and test sets."""
    print("\n=== Splitting Data ===")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    print("\n=== Scaling Features ===")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled using StandardScaler.")
    
    return X_train_scaled, X_test_scaled, scaler

def save_outputs(X_train, X_test, y_train, y_test, scaler, encoders, 
                 feature_names, genre_encoder, genres_discarded, balancing_strategy, output_dir):
    """Save all preprocessed data, models, and metadata."""
    print("\n=== Saving Outputs ===")
    os.makedirs(output_dir, exist_ok=True)
    
    pd.DataFrame(X_train, columns=feature_names).to_csv(f'{output_dir}/X_train.csv', index=False)
    pd.DataFrame(X_test, columns=feature_names).to_csv(f'{output_dir}/X_test.csv', index=False)
    pd.DataFrame(y_train, columns=['genre_encoded']).to_csv(f'{output_dir}/y_train.csv', index=False)
    pd.DataFrame(y_test, columns=['genre_encoded']).to_csv(f'{output_dir}/y_test.csv', index=False)
    print(f"Saved train/test CSVs to {output_dir}/")
    
    with open(f'{output_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved scaler.pkl")
    
    encoders['genre'] = genre_encoder
    with open(f'{output_dir}/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    print("Saved encoders.pkl")
    
    metadata = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'n_genres': len(genre_encoder.classes_),
        'genre_labels': {i: label for i, label in enumerate(genre_encoder.classes_)},
        'balancing_strategy': balancing_strategy,
        'genres_discarded': genres_discarded,
        'random_state': RANDOM_STATE
    }
    
    with open(f'{output_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata.json")

def print_summary(metadata_path):
    """Print a final summary of the preprocessing results."""
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\nFinal Dataset Summary:")
    print(f"  - Balancing Strategy: {metadata['balancing_strategy']}")
    print(f"  - Genres Discarded: {metadata['genres_discarded']}")
    print(f"  - Features: {metadata['n_features']}")
    print(f"  - Training samples: {metadata['n_train_samples']}")
    print(f"  - Test samples: {metadata['n_test_samples']}")
    print(f"  - Final Genres: {metadata['n_genres']}")
    
    print(f"\nFinal Genre Labels: {', '.join(metadata['genre_labels'].values())}")
    print(f"\nFeatures: {', '.join(metadata['feature_names'])}")
    print(f"\nAll outputs saved to '{OUTPUT_DIR}/' directory.")
    print("Team members can now use the CSV files in this directory for modeling.")
    print("\n" + "="*60)

def main():
    """Main preprocessing pipeline."""
    print("="*60)
    print("SPOTIFY GENRE CLASSIFICATION - DATA PREPROCESSING")
    print("="*60)
    
    df = load_data('SpotifyFeatures.csv')
    df = clean_data(df)
    df, genres_discarded = engineer_genres(df)
    
    explore_data(df) 
    
    df = balance_data_hybrid(df, sample_size=SAMPLE_SIZE, random_state=RANDOM_STATE)
    
    # This function now correctly handles time_signature
    df, encoders = encode_categorical(df)
    
    X, y, feature_names, genre_encoder = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=RANDOM_STATE)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    balancing_strategy_desc = (
        f"Hybrid: Undersample majority classes to {SAMPLE_SIZE}, "
        f"keep all samples for classes with <= {SAMPLE_SIZE}."
    )
    
    save_outputs(X_train_scaled, X_test_scaled, y_train, y_test, 
                 scaler, encoders, feature_names, genre_encoder, genres_discarded,
                 balancing_strategy_desc, output_dir=OUTPUT_DIR)
    
    print_summary(f'{OUTPUT_DIR}/metadata.json')

if __name__ == "__main__":
    main()

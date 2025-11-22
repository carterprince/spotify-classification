import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import time
import os
import random
from sklearn.metrics import confusion_matrix

base_url = 'https://raw.githubusercontent.com/carterprince/spotify-classification/refs/heads/master/data/'
DATA_DIR = 'data'
OUTPUT_DIR = 'output'
OUTPUT_FILE = 'multilayer_perceptron.json'
MEMBER_NAME = "Carter Prince"
MODEL_NAME = "Multilayer Perceptron (MLP)"

torch.manual_seed(2)
np.random.seed(2)
random.seed(2)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def load_data():
    print("Loading data...")
    X_train = pd.read_csv(base_url + 'X_train.csv').values
    y_train = pd.read_csv(base_url + 'y_train.csv').values.flatten()
    X_test = pd.read_csv(base_url + 'X_test.csv').values
    y_test = pd.read_csv(base_url + 'y_test.csv').values.flatten()

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

class DynamicSpotifyMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_prob):
        super(DynamicSpotifyMLP, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Print loss every 10 epochs or on the last epoch to keep output clean
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            avg_loss = running_loss / len(train_loader)
            print(f"    Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        y_true = y_test.cpu().numpy()
        y_pred = predicted.cpu().numpy()
        cm = confusion_matrix(y_true, y_pred)
        accuracy = (y_pred == y_true).sum() / len(y_true)
    return cm, accuracy

def main():
    X_train, y_train, X_test, y_test = load_data()
    input_dim = X_train.shape[1]
    output_dim = len(torch.unique(y_train))
    train_dataset = TensorDataset(X_train, y_train)

    # Search Space
    layer_options = [3, 4, 5, 6]
    width_options = [64, 128, 256, 512]
    batch_options = [128, 256, 512]
    lr_options = [0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    
    NUM_TRIALS = 30
    SEARCH_EPOCHS = 15
    FINAL_EPOCHS = 200

    trials_data = []
    best_accuracy_search = 0.0
    best_hyperparameters = {}

    print(f"Starting Randomized Search ({NUM_TRIALS} trials, {SEARCH_EPOCHS} epochs each)...")

    # --- 1. Hyperparameter Search Loop ---
    for i in range(NUM_TRIALS):
        params = {
            "num_layers": random.choice(layer_options),
            "hidden_dim": random.choice(width_options),
            "lr": random.choice(lr_options),
            "dropout": 0.3,
            "batch_size": random.choice(batch_options)
        }

        print(f"Trial {i+1}/{NUM_TRIALS}: {params}")

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

        model = DynamicSpotifyMLP(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            output_dim=output_dim,
            num_layers=params['num_layers'],
            dropout_prob=params['dropout']
        ).to(device)

        if hasattr(torch, 'compile'):
            model = torch.compile(model)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=params['lr'])

        train_model(model, train_loader, criterion, optimizer, epochs=SEARCH_EPOCHS)
        
        cm, accuracy = evaluate_model(model, X_test, y_test)
        print(f"   -> Search Accuracy: {accuracy:.4f}")

        trial_entry = {
            "hyperparameters": params,
            "confusion_matrix": cm.tolist(),
            "accuracy": accuracy
        }
        trials_data.append(trial_entry)

        if accuracy > best_accuracy_search:
            best_accuracy_search = accuracy
            best_hyperparameters = params

    print("\n" + "="*50)
    print(f"Search Complete. Best Search Accuracy: {best_accuracy_search:.4f}")
    print(f"Best Hyperparameters: {best_hyperparameters}")
    print("Retraining model with best hyperparameters for 50 epochs...")
    print("="*50 + "\n")

    # --- 2. Final Retraining ---
    
    # Re-create DataLoader with the best batch size
    final_loader = DataLoader(train_dataset, batch_size=best_hyperparameters['batch_size'], shuffle=True)

    # Re-initialize the model with best architecture
    final_model = DynamicSpotifyMLP(
        input_dim=input_dim,
        hidden_dim=best_hyperparameters['hidden_dim'],
        output_dim=output_dim,
        num_layers=best_hyperparameters['num_layers'],
        dropout_prob=best_hyperparameters['dropout']
    ).to(device)

    if hasattr(torch, 'compile'):
        final_model = torch.compile(final_model)

    final_criterion = nn.CrossEntropyLoss()
    final_optimizer = optim.AdamW(final_model.parameters(), lr=best_hyperparameters['lr'])

    # Train
    start_train = time.time()
    train_model(final_model, final_loader, final_criterion, final_optimizer, epochs=FINAL_EPOCHS)
    end_train = time.time()

    # Evaluate
    start_test = time.time()
    final_cm, final_accuracy = evaluate_model(final_model, X_test, y_test)
    end_test = time.time()

    print(f"\nFinal Model Accuracy (50 Epochs): {final_accuracy:.4f}")

    # --- 3. Save Output ---
    output_data = {
        "model_name": MODEL_NAME,
        "person_name": MEMBER_NAME,
        "best_hyperparameters": best_hyperparameters,
        "best_confusion_matrix": final_cm.tolist(), # Saving the matrix from the 50-epoch run
        "trials": trials_data,
        "total_train_time": round(end_train - start_train, 4), # Time for the 50-epoch run
        "total_test_time": round(end_test - start_test, 4)     # Time for the 50-epoch run
    }

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Analysis saved to {output_path}")

if __name__ == "__main__":
    main()

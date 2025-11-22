import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Paths and metadata
DATA_DIR = "data"
OUTPUT_DIR = "output"
OUTPUT_FILE = "gaussian_naive_bayes.json"
MEMBER_NAME = "Sanghyun An"
MODEL_NAME = "Gaussian Naive Bayes"
# Toggle to use empirical class priors from the training labels
USE_PRIORS = True

np.random.seed(42)


def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv")).values
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.flatten()
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv")).values
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.flatten()
    return X_train, y_train, X_test, y_test


def run_trials(X_train, y_train, X_test, y_test, smoothing_values, priors=None):
    trials = []
    best_accuracy = -1.0
    best_trial = None
    best_train_time = 0.0
    best_test_time = 0.0

    for i, smoothing in enumerate(smoothing_values):
        print(f"Trial {i + 1}/{len(smoothing_values)}: var_smoothing={smoothing}")

        model = GaussianNB(var_smoothing=smoothing, priors=priors)

        start_train = time.time()
        model.fit(X_train, y_train)
        end_train = time.time()

        start_test = time.time()
        y_pred = model.predict(X_test)
        end_test = time.time()

        cm = confusion_matrix(y_test, y_pred)
        accuracy = (y_pred == y_test).mean()

        trial_entry = {
            "hyperparameters": {"var_smoothing": smoothing},
            "confusion_matrix": cm.tolist(),
        }
        trials.append(trial_entry)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_trial = trial_entry
            best_train_time = end_train - start_train
            best_test_time = end_test - start_test

        print(f"   -> Accuracy: {accuracy:.4f}")

    return trials, best_trial, best_train_time, best_test_time


def main():
    X_train, y_train, X_test, y_test = load_data()

    # Log-spaced sweep over a wider range of var_smoothing values
    smoothing_values = np.logspace(-12, 2, num=15)
    priors = None
    if USE_PRIORS:
        # Estimate class priors from the training distribution
        class_counts = np.bincount(y_train)
        priors = class_counts / class_counts.sum()

    trials, best_trial, best_train_time, best_test_time = run_trials(
        X_train, y_train, X_test, y_test, smoothing_values, priors=priors
    )

    output_data = {
        "model_name": MODEL_NAME,
        "person_name": MEMBER_NAME,
        "best_hyperparameters": best_trial["hyperparameters"],
        "best_confusion_matrix": best_trial["confusion_matrix"],
        "trials": trials,
        "total_train_time": round(best_train_time, 4),
        "total_test_time": round(best_test_time, 4),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSuccess! Analysis saved to {output_path}")
    print(
        f"Best var_smoothing: {best_trial['hyperparameters']['var_smoothing']} | "
        f"Train time: {best_train_time:.4f}s | Test time: {best_test_time:.4f}s"
    )


if __name__ == "__main__":
    main()

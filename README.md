# spotify-classification

Term Project for Data Science II (Group 8)

A comparative analysis of six machine learning models for predicting the genre of songs on Spotify based on their audio features and metadata.

Members:

- Robera Abajobir
- Sanghyun An
- Austin Bell
- Carter Prince
- Tyler Varma
- Anvita Yerramsetty

## === Info for Group Members: ===

**Files you need:**

In the `preprocessed_data/` directory:

- `X_train.csv`
- `X_test.csv`
- `y_train.csv`
- `y_test.csv`

These contain your preprocessed data. Train and test on these.

Note: You do not need to scale, clean, or split the data, this is already done for you.

**Files you can ignore:**

`metadata.json` maps integer labels to genre names and feature info. This file will be needed in the final analysis notebook, but each individual won't need to use it.

`scaler.pkl` is saved "just in case" but again you shouldn't need it. It lets you reverse the scaling from the preprocessing script or scale new data the same way.

**Deliverables for the final analysis notebook:**

Have your notebook write a single JSON file to the `output/` folder. You can call it whatever you want, but just going with the model name in snake case (e.g. `random_forest.json`) is probably best.

This JSON file is the only actual output required from your code. This ensures that for each model we have a consistent, streamlined way to read its performance and to see how each hyperparameter affected its performance. This will be immensely useful for writing the final report.

Use this format for the JSON (these are just example values):

```json
{
  "model_name": "Random Forest",
  "person_name": "Anvita Yerramsetty",
  "best_hyperparameters": {
    "n_estimators": 200,
    "max_depth": 25,
    "min_samples_split": 5,
    "min_samples_leaf": 2
    ...
  },
  "best_confusion_matrix": [
    [145, 12, 3, 8, 2, ...],
    [10, 167, 5, 4, 1, ...],
    [5, 8, 152, 15, 3, ...],
    ...
  ],
  "trials": [
    {
      "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1
      },
      "confusion_matrix": [
        [142, 15, 4, 9, 3, ...],
        [12, 162, 7, 5, 2, ...],
        ...
      ]
    },
    {
      "hyperparameters": {
        "n_estimators": 200,
        "max_depth": 25,
        "min_samples_split": 5,
        "min_samples_leaf": 2
      },
      "confusion_matrix": [
        [145, 12, 3, 8, 2, ...],
        [10, 167, 5, 4, 1, ...],
        ...
      ]
    },
    ...
  ],
  "total_train_time": 127.45,
  "total_test_time": 33.45
}
```

## Preliminary Test

Here's a performance baseline I got using a simple KNN model, which suggests there's strong signal present in the data:

![](/knn_confusion_matrix.png)

## Requirements

## Run

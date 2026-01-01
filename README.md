# Spotify Genre Classification
**Term Project for Data Science II (Group 8)**

[**📄 Click here to view the Full Project Report**](https://raw.githubusercontent.com/carterprince/spotify-classification/refs/heads/master/report.pdf)

A comparative analysis of six machine learning models for predicting the genre of songs on Spotify based on their audio features and metadata.

## Group Members
*   Anvita Yerramsetty
*   Austin Bell
*   Carter Prince
*   Robera Abajobir
*   Sanghyun An
*   Tyler Varma

## Project Overview
This project aims to classify music tracks into 24 distinct genres using 14 audio features extracted from the Spotify API. We implemented a complete data science pipeline including data cleaning, hybrid balancing, feature scaling, hyperparameter tuning, and comparative analysis.

The models evaluated are:
1.  **Logistic Regression**
2.  **K-Nearest Neighbors (KNN)**
3.  **Gaussian Naive Bayes**
4.  **Random Forest Classifier**
5.  **Gradient Boosting (XGBoost)**
6.  **Multilayer Perceptron (Neural Network)**

## Directory Structure

```text
.
├── data/                   # Contains processed CSV files and metadata
├── output/                 # JSON files containing results from each model run
├── preprocess.py           # Script to clean, balance, and scale the raw data
├── analyze.py              # Script to generate the results table and leaderboard
├── generate_figures.py     # Script to create all visualizations for the report
├── report.tex              # Final LaTeX report source code
├── report.pdf              # Final report PDF
├── requirements.txt        # Python dependencies
└── [model_scripts].py      # Individual training scripts (e.g., xgboost.ipynb, mlp.py)
```

## How to Reproduce Results

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install pandas numpy scikit-learn xgboost torch matplotlib seaborn
```

### 2. Data Preprocessing
The raw data (`SpotifyFeatures.csv`) is processed into training and testing sets. This script handles cleaning, encoding, scaling, and splitting.

```bash
python preprocess.py
```
*Output: Generates `X_train.csv`, `y_train.csv`, `X_test.csv`, `y_test.csv` in the `data/` folder.*

### 3. Model Training
Each model has its own script or notebook. Running these will perform hyperparameter tuning and save the results to a standardized JSON file in the `output/` directory.

*Example:*
```bash
python mlp.py
python gaussian_nb.py
# ... etc
```

*Note: The `output/` directory already contains the results from our final runs, so re-training is optional.*

### 4. Analysis & Visualization
To generate the comparative leaderboard and the figures used in the report:

```bash
# Generates the leaderboard and efficiency plots
python analyze.py

# Generates specific figures (Class Balance, Correlation, Hyperparameter plots)
python generate_figures.py
```

"""
Day 1: Minimal Baseline on Iris Dataset
======================================

This script demonstrates a simple machine learning workflow using the classic Iris dataset.
It covers the following steps:
    1. Data loading and inspection
    2. Train/test splitting with stratification
    3. Quick exploratory data analysis (EDA)
    4. Model building using pipelines (Logistic Regression, Decision Tree, Random Forest)
    5. Model training and evaluation
    6. Confusion matrix visualization
    7. Cross-validation for Random Forest
    8. Summary of results

Main Components:
----------------
- DataBundle: A dataclass for storing train/test splits and metadata.
- load_data: Loads the Iris dataset and splits it into train/test sets.
- build_logreg_pipeline: Creates a pipeline for logistic regression with scaling.
- build_tree_pipeline: Creates a pipeline for decision tree (no scaling needed).
- build_rf_pipeline: Creates a pipeline for random forest (no scaling needed).
- train_and_eval: Trains a model, evaluates accuracy, and prints a classification report.
- plot_cm: Plots the confusion matrix for a given model.
- quick_eda: Displays a quick summary and statistics of the feature data.
- main: Orchestrates the workflow, including cross-validation for Random Forest.

Usage:
------
Run this script directly to see printed outputs and confusion matrix plots.
You can modify or extend the pipelines and evaluation steps for experimentation.
Cross-validation results for Random Forest are also printed for deeper model assessment.

Author: [Your Name]
Date: 2025-09-05
"""

# Goal: load -> split -> train -> evaluate -> quick plot

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

RANDOM_SEED = 42

@dataclass
class DataBundle:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list
    target_names: list

def load_data(test_size: float = 0.2) -> DataBundle:
    iris = load_iris()
    # Visualize iris object structure
    print("\n=== iris object ===")
    print(f"Type: {type(iris)}")
    print(f"Keys: {iris.keys()}")
    print(f"Feature names: {iris.feature_names}")
    print(f"Target names: {iris.target_names}")
    print(f"Data shape: {iris.data.shape}")
    print(f"Target shape: {iris.target.shape}")
    print(f"First 5 rows of data:\n{iris.data[:5]}")
    print(f"First 5 targets: {iris.target[:5]}")
    # X contains all the measurements (features)
    X = iris.data  # Shape: (150, 4) - 150 flowers, 4 measurements each
    print("Features we measure:", iris.feature_names)
    # ['sepal length', 'sepal width', 'petal length', 'petal width']
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )
    return DataBundle(X_train, X_test, y_train, y_test, list(iris.feature_names), list(iris.target_names))

def build_logreg_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()), # A scaler is used to make the dataset scaled and fair.
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)) #clf stands for classifier
    ])

def build_tree_pipeline() -> Pipeline:
    # Tree doesn’t need scaling, but we keep Pipeline for symmetry/consistency
    return Pipeline([
        ("clf", DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED))
    ])

def build_rf_pipeline() -> Pipeline:
    # RF is tree-based → no scaling required
    return Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=300,      # number of trees
            max_depth=5,           # let trees grow; can tune later
            min_samples_leaf=2,    # avoid overfitting (corrected argument)
            random_state=RANDOM_SEED,
            n_jobs=-1              # use all cores
        ))
    ])

def train_and_eval(model: Pipeline, data: DataBundle, label: str) -> Tuple[float, str, np.ndarray]:
    model.fit(data.X_train, data.y_train)
    preds = model.predict(data.X_test)
    acc = accuracy_score(data.y_test, preds)
    report = classification_report(data.y_test, preds, target_names=data.target_names)
    cm = confusion_matrix(data.y_test, preds)
    print(f"\n=== {label} ===")
    print(f"Accuracy: {acc:.3f}")
    print(report)
    return acc, label, cm

def plot_cm(cm: np.ndarray, labels: list, title: str):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def quick_eda(data: DataBundle):
    """
    Quick Exploratory Data Analysis (EDA)

    Combines training and test feature data into a single DataFrame,
    prints the first few rows, and displays summary statistics for each feature.
    This helps you rapidly inspect the structure and distribution of the dataset
    before modeling.
    """
    df = pd.DataFrame(
        np.vstack([data.X_train, data.X_test]),
        columns=data.feature_names
    )
    print("\n=== Quick EDA (head) ===")
    print(df.head())
    print("\nDescribe:")
    print(df.describe())

def main():
    """
    Main function to run baseline classification models on the Iris dataset.
    Workflow:
    1. Loads and splits the Iris dataset.
    2. Performs quick exploratory data analysis (EDA).
    3. Builds three classification pipelines: Logistic Regression, Decision Tree, and Random Forest.
    4. Evaluates Random Forest using 5-fold cross-validation:
        - Cross-validation provides a robust estimate of model performance by splitting the training data into multiple folds,
          training and validating the model on each fold, and reporting the mean and standard deviation of accuracy scores.
          This helps assess the model's generalizability and reduces the risk of overfitting to a single train/test split.
    5. Trains and evaluates all models on the held-out test set.
    6. Plots confusion matrices for each model.
    7. Prints a summary of test accuracies and identifies the best-performing baseline model.
    Returns:
        None
    """
    data = load_data(test_size=0.2)
    quick_eda(data)

    logreg = build_logreg_pipeline()
    tree = build_tree_pipeline()
    rf = build_rf_pipeline()

    # --- NEW: Cross-validation for RF ---

    print("\n=== Random Forest Cross-Validation ===")
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(rf, data.X_train, data.y_train, cv=5)
    print("Cross-val scores:", scores)
    print("Mean CV accuracy:", scores.mean())
    print("Std deviation:", scores.std())

    # -----------------------------------

    acc1, label1, cm1 = train_and_eval(logreg, data, "Logistic Regression")
    acc2, label2, cm2 = train_and_eval(tree, data, "Decision Tree")
    acc3, label3, cm3 = train_and_eval(rf, data, "Random Forest")


    # Plot confusion matrices (one by one)
    plot_cm(cm1, data.target_names, f"{label1} — Confusion Matrix")
    plot_cm(cm2, data.target_names, f"{label2} — Confusion Matrix")
    plot_cm(cm3, data.target_names, f"{label3} — Confusion Matrix")

    # Simple result summary
    print("\n=== Summary ===")
    print(f"{label1} Acc: {acc1:.3f}")
    print(f"{label2} Acc: {acc2:.3f}")
    print(f"{label3} Acc: {acc3:.3f}")
    winner = max([(acc1, label1), (acc2, label2), (acc3, label3)], key=lambda x: x[0])[1]
    print(f"Baseline winner: {winner}")

if __name__ == "__main__":
    main()

"""
Day 2: Minimal Baseline on Breast Cancer Dataset
===============================================

This script demonstrates a simple machine learning workflow using the classic Breast Cancer dataset.
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
- load_data: Loads the Breast Cancer dataset and splits it into train/test sets.
- build_logreg_pipeline: Creates a pipeline for logistic regression with scaling.
- build_tree_pipeline: Creates a pipeline for decision tree (no scaling needed).
- build_rf_pipeline: Creates a pipeline for random forest (no scaling needed).
- train_and_eval: Trains a model, evaluates accuracy, and prints a classification report.
- plot_cm: Plots the confusion matrix for a given model, with optional accuracy in the title.
- quick_eda: Displays a quick summary and statistics of the feature data, including head and describe.
- main: Orchestrates the workflow, including cross-validation for Random Forest.

Usage:
------
Run this script directly to see printed outputs and confusion matrix plots.
You can modify or extend the pipelines and evaluation steps for experimentation.

Author: Prasanna Subash
Date: 2025-09-05
"""

# Goal: load -> split -> train -> evaluate -> quick plot

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
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
    breast_cancer = load_breast_cancer()
    # Visualize breast_cancer object structure
    print("\n=== breast_cancer object ===")
    print(f"Type: {type(breast_cancer)}")
    print(f"Keys: {breast_cancer.keys()}")
    print(f"Feature names: {breast_cancer.feature_names}")
    print(f"Target names: {breast_cancer.target_names}")
    print(f"Data shape: {breast_cancer.data.shape}")
    print(f"Target shape: {breast_cancer.target.shape}")
    print(f"First 5 rows of data:\n{breast_cancer.data[:5]}")
    print(f"First 5 targets: {breast_cancer.target[:5]}")
    X = breast_cancer.data
    y = breast_cancer.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )
    return DataBundle(X_train, X_test, y_train, y_test, list(breast_cancer.feature_names), list(breast_cancer.target_names))

def build_logreg_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
    ])

def build_tree_pipeline() -> Pipeline:
    # Tree doesn’t need scaling, but we keep Pipeline for symmetry/consistency
    return Pipeline([
        ("clf", DecisionTreeClassifier(random_state=RANDOM_SEED))
    ])

def build_rf_pipeline() -> Pipeline:
    # Random Forest does not need scaling
    return Pipeline([
        ("clf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED))
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

def plot_cm(cm: np.ndarray, labels: list, title: str, accuracy: float = None):
    """
    Plots the confusion matrix for a given model.
    Optionally displays accuracy in the plot title.
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(values_format="d")  # nice blue shading
    
    if accuracy is not None:
        plt.title(f"{title}\nAccuracy: {accuracy*100:.1f}%")
    else:
        plt.title(title)
    
    plt.tight_layout()
    plt.show()

def quick_eda(data: DataBundle):
    """
    Quick exploratory data analysis:
    - Shows the head of the combined train/test data as a DataFrame
    - Prints summary statistics (describe)
    - Useful for sanity-checking feature distributions and spotting outliers
    """
    df = pd.DataFrame(
        np.vstack([data.X_train, data.X_test]),
        columns=data.feature_names
    )
    print("\n=== Quick EDA (head) ===")
    print(df.head())
    print("\nDescribe:")
    print(df.describe())

def cv_report(name, model, data):
    """
    Generates and prints a cross-validation report for a given model and dataset.

    Parameters:
        name (str): The name of the model or estimator to display in the report.
        model (estimator): The machine learning model to evaluate.
        data (object): An object containing training data with attributes 'X_train' and 'y_train'.

    Returns:
        tuple: A tuple containing the mean and standard deviation of the cross-validation scores.

    Prints:
        A formatted string displaying the model name, mean cross-validation score, and its standard deviation.
    """
    scores = cross_val_score(model, data.X_train, data.y_train, cv=5)
    print(f"{name:>14}  CV mean={scores.mean():.4f}  ±{scores.std():.4f}")
    return scores.mean(), scores.std()

def main():
    """
    Main workflow:
    - Loads and splits data
    - Runs quick EDA
    - Builds and trains three models (Logistic Regression, Decision Tree, Random Forest)
    - Evaluates models and prints classification reports
    - Performs cross-validation for Random Forest
    - Plots confusion matrices with accuracy
    - Prints summary of results
    """
    data = load_data(test_size=0.2)
    quick_eda(data)

    logreg = build_logreg_pipeline()
    tree = build_tree_pipeline()
    rf = build_rf_pipeline()  # Add Random Forest

    acc1, label1, cm1 = train_and_eval(logreg, data, "Logistic Regression")
    acc2, label2, cm2 = train_and_eval(tree, data, "Decision Tree")
    acc3, label3, cm3 = train_and_eval(rf, data, "Random Forest")  # Train and evaluate RF

    # Cross-validation for the 3 models
    cv_report("LogReg", logreg, data)
    cv_report("DecisionTree", tree, data)
    cv_report("RandomForest", rf, data)

    # Plot confusion matrices (one by one)
    plot_cm(cm1, data.target_names, f"{label1} — Confusion Matrix", acc1)
    plot_cm(cm2, data.target_names, f"{label2} — Confusion Matrix", acc2)
    plot_cm(cm3, data.target_names, f"{label3} — Confusion Matrix", acc3)  # Plot RF CM

    # Simple result summary
    print("\n=== Summary ===")
    print(f"{label1} Acc: {acc1:.3f}")
    print(f"{label2} Acc: {acc2:.3f}")
    print(f"{label3} Acc: {acc3:.3f}")

if __name__ == "__main__":
    main()

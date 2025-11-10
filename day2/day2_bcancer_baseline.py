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
    7. ROC-AUC curve plotting for selected models
    8. Cross-validation for all models
    9. Summary of results

Main Components:
----------------
- DataBundle: A dataclass for storing train/test splits and metadata.
- load_data: Loads the Breast Cancer dataset and splits it into train/test sets.
- build_logreg_pipeline: Creates a pipeline for logistic regression with scaling.
- build_tree_pipeline: Creates a pipeline for decision tree (no scaling needed).
- build_rf_pipeline: Creates a pipeline for random forest (no scaling needed).
- train_and_eval: Trains a model, evaluates accuracy, and prints a classification report.
- plot_cm: Plots the confusion matrix for a given model, with optional accuracy in the title.
- plot_roc: Plots the ROC curve and computes AUC for a given model.
- quick_eda: Displays a quick summary and statistics of the feature data, including head and describe.
- cv_report: Prints cross-validation results for a given model.
- evaluate_threshold: Computes confusion-matrix stats for a given threshold.
- threshold_sweep: Performs threshold tuning to balance FP/FN trade-offs.
- main: Orchestrates the workflow, including cross-validation and ROC plotting.

Usage:
------
Run this script directly to see printed outputs and confusion matrix/ROC plots.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 42


@dataclass
class DataBundle:
    """Container for train/test splits and metadata."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list
    target_names: list


def load_data(test_size: float = 0.2) -> DataBundle:
    """
    Load and split the breast cancer dataset.
    
    Args:
        test_size: Proportion of dataset to include in the test split.
        
    Returns:
        DataBundle containing train/test splits and metadata.
    """
    breast_cancer = load_breast_cancer()
    
    # Visualize breast_cancer object structure
    print("\n=== Breast Cancer Dataset Overview ===")
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
    
    return DataBundle(
        X_train, X_test, y_train, y_test, 
        list(breast_cancer.feature_names), 
        list(breast_cancer.target_names)
    )


def build_logreg_pipeline() -> Pipeline:
    """Build logistic regression pipeline with scaling."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
    ])


def build_tree_pipeline() -> Pipeline:
    """Build decision tree pipeline (no scaling needed)."""
    return Pipeline([
        ("clf", DecisionTreeClassifier(random_state=RANDOM_SEED))
    ])


def build_rf_pipeline() -> Pipeline:
    """Build random forest pipeline (no scaling needed)."""
    return Pipeline([
        ("clf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED))
    ])


def train_and_eval(model: Pipeline, data: DataBundle, label: str) -> Tuple[float, str, np.ndarray]:
    """
    Train and evaluate a model.
    
    Args:
        model: The pipeline to train.
        data: DataBundle containing train/test data.
        label: Model name for display.
        
    Returns:
        Tuple of (accuracy, label, confusion_matrix).
    """
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
    Plot confusion matrix with optional accuracy display.
    
    Args:
        cm: Confusion matrix array.
        labels: Target class labels.
        title: Plot title.
        accuracy: Optional accuracy to display in title.
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(values_format="d")
    
    if accuracy is not None:
        plt.title(f"{title}\nAccuracy: {accuracy*100:.1f}%")
    else:
        plt.title(title)
    
    plt.tight_layout()
    plt.show()


def plot_roc(model, X_test, y_test, title):
    """
    Plot ROC curve and compute AUC score.

    Args:
        model: Trained model with predict_proba or decision_function.
        X_test: Test feature data.
        y_test: True labels for test data.
        title: Plot title.

    Returns:
        float: The computed AUC score.

    The function will:
    - Use predict_proba (preferred) or decision_function to get scores.
    - Compute false positive rate (FPR), true positive rate (TPR), and AUC.
    - Plot the ROC curve with AUC in the legend.
    - Show the plot.
    """
    # Get prediction probabilities
    proba = (model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") 
             else model.decision_function(X_test))
    
    fpr, tpr, _ = roc_curve(y_test, proba)
    score = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return score


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


def cv_report(name: str, model, data: DataBundle) -> Tuple[float, float]:
    """
    Generate and print cross-validation report.

    Args:
        name: Model name for display.
        model: The machine learning model to evaluate.
        data: DataBundle containing training data.

    Returns:
        Tuple of (mean_cv_score, std_cv_score).
    """
    scores = cross_val_score(model, data.X_train, data.y_train, cv=5)
    print(f"{name:>14}  CV mean={scores.mean():.4f}  ±{scores.std():.4f}")
    return scores.mean(), scores.std()


def evaluate_threshold(model, X, y, threshold: float) -> dict:
    """
    Compute confusion-matrix stats for a given threshold.
    
    Args:
        model: Trained model with predict_proba.
        X: Feature data.
        y: True labels.
        threshold: Classification threshold.
        
    Returns:
        Dictionary with threshold metrics.
    """
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / np.sum(cm)
    
    return {
        "threshold": threshold,
        "accuracy": acc,
        "false_negatives": fn,
        "false_positives": fp,
        "true_positives": tp,
        "true_negatives": tn,
    }


def threshold_sweep(model, X, y) -> dict:
    """
    Perform threshold sweep to find optimal classification threshold.
    
    Args:
        model: Trained model with predict_proba.
        X: Feature data.
        y: True labels.
        
    Returns:
        Dictionary with best threshold metrics.
    """
    thresholds = np.linspace(0.2, 0.8, 13)  # Test thresholds between 0.2–0.8
    results = [evaluate_threshold(model, X, y, t) for t in thresholds]
    
    print("\n=== Threshold Sweep (Random Forest) ===")
    for r in results:
        print(f"thr={r['threshold']:.2f} | acc={r['accuracy']:.3f} | "
              f"FN={r['false_negatives']} | FP={r['false_positives']}")
    
    # Find best threshold (minimize false negatives, then maximize accuracy)
    best = min(results, key=lambda x: (x["false_negatives"], -x["accuracy"]))
    print(f"\nBest threshold: {best['threshold']:.2f} "
          f"(FN={best['false_negatives']}, FP={best['false_positives']}, "
          f"acc={best['accuracy']:.3f})")
    
    return best


def main():
    """
    Main workflow:
    - Loads and splits data
    - Runs quick EDA
    - Builds and trains three models (Logistic Regression, Decision Tree, Random Forest)
    - Evaluates models and prints classification reports
    - Performs cross-validation for all models
    - Plots confusion matrices with accuracy
    - Plots ROC curves for selected models
    - Performs threshold tuning
    - Prints summary of results
    """
    # Load and explore data
    data = load_data(test_size=0.2)
    quick_eda(data)

    # Build model pipelines
    logreg = build_logreg_pipeline()
    tree = build_tree_pipeline()
    rf = build_rf_pipeline()

    # Train and evaluate models
    acc1, label1, cm1 = train_and_eval(logreg, data, "Logistic Regression")
    acc2, label2, cm2 = train_and_eval(tree, data, "Decision Tree")
    acc3, label3, cm3 = train_and_eval(rf, data, "Random Forest")

    # Cross-validation for all models
    print("\n=== Cross-Validation Results ===")
    cv_report("LogReg", logreg, data)
    cv_report("DecisionTree", tree, data)
    cv_report("RandomForest", rf, data)

    # Plot confusion matrices
    plot_cm(cm1, data.target_names, f"{label1} — Confusion Matrix", acc1)
    plot_cm(cm2, data.target_names, f"{label2} — Confusion Matrix", acc2)
    plot_cm(cm3, data.target_names, f"{label3} — Confusion Matrix", acc3)

    # Plot ROC curves for selected models
    auc_lr = plot_roc(logreg, data.X_test, data.y_test, "LogReg ROC")
    auc_rf = plot_roc(rf, data.X_test, data.y_test, "RandomForest ROC")

    # Threshold tuning for Random Forest
    threshold_sweep(rf, data.X_test, data.y_test)

    # Summary of results
    print("\n=== Final Summary ===")
    print(f"{label1} Acc: {acc1:.3f} | AUC: {auc_lr:.3f}")
    print(f"{label2} Acc: {acc2:.3f}")
    print(f"{label3} Acc: {acc3:.3f} | AUC: {auc_rf:.3f}")


if __name__ == "__main__":
    main()

"""
Day 3: Model Tuning & Hyperparameter Optimization on Breast Cancer Dataset
===========================================================================

This script demonstrates advanced machine learning techniques including hyperparameter tuning
and model optimization using the Breast Cancer dataset. It builds on baseline models
by implementing GridSearchCV, ROC/AUC analysis, and model persistence.

It covers the following steps:
    1. Data loading and inspection
    2. Train/test splitting with stratification
    3. Quick exploratory data analysis (EDA)
    4. Model building using pipelines (Logistic Regression, Decision Tree, Random Forest)
    5. Cross-validation for baseline comparison
    6. Model training and evaluation on holdout test set
    7. Confusion matrix visualization
    8. ROC-AUC curve plotting for model comparison
    9. GridSearchCV for Random Forest hyperparameter tuning
    10. Model persistence (saving tuned model with joblib)
    11. Comprehensive results summary

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
- main: Orchestrates the complete workflow including GridSearchCV tuning and model saving.

Usage:
------
Run this script directly to see printed outputs, confusion matrix/ROC plots, and GridSearchCV results.
The best tuned model will be saved to models/best_rf_bcancer.joblib.

Author: Prasanna Subash
Date: 2025-11-10
"""

# Goal: load -> split -> train -> evaluate -> quick plot

from dataclasses import dataclass
from typing import Tuple
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
import joblib
import os

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
s
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


# def threshold_sweep(model, X, y) -> dict:
#     """
#     Perform threshold sweep to find optimal classification threshold.
    
#     Args:
#         model: Trained model with predict_proba.
#         X: Feature data.
#         y: True labels.
        
#     Returns:
#         Dictionary with best threshold metrics.
#     """
#     thresholds = np.linspace(0.2, 0.8, 13)  # Test thresholds between 0.2–0.8
#     results = [evaluate_threshold(model, X, y, t) for t in thresholds]
    
#     print("\n=== Threshold Sweep (Random Forest) ===")
#     for r in results:
#         print(f"thr={r['threshold']:.2f} | acc={r['accuracy']:.3f} | "
#               f"FN={r['false_negatives']} | FP={r['false_positives']}")
    
#     # Find best threshold (minimize false negatives, then maximize accuracy)
#     best = min(results, key=lambda x: (x["false_negatives"], -x["accuracy"]))
#     print(f"\nBest threshold: {best['threshold']:.2f} "
#           f"(FN={best['false_negatives']}, FP={best['false_positives']}, "
#           f"acc={best['accuracy']:.3f})")
    
#     return best





def main():
    """
    Main orchestration for Day 3:
    - load data
    - quick EDA
    - build models
    - cross-validate each model on the TRAIN split
    - train on the train split and evaluate on the holdout test set
    - plot ROC curves
    - run a single GridSearchCV for RandomForest (small grid)
    - evaluate tuned RF and save it
    - print final summary
    """
    # 1) Load data + quick EDA
    data = load_data(test_size=0.2)
    quick_eda(data)

    # 2) Build model pipelines
    logreg = build_logreg_pipeline()
    tree = build_tree_pipeline()
    rf = build_rf_pipeline()   # must be a Pipeline with final step named "clf"

    # 3) Cross-validation report (on TRAIN SPLIT only)
    from sklearn.model_selection import cross_val_score

    def cv_report(name, model, data, cv=5):
        scores = cross_val_score(model, data.X_train, data.y_train, cv=cv, n_jobs=-1)
        print(f"{name:>14}  CV mean={scores.mean():.4f}  ±{scores.std():.4f}")
        return scores.mean(), scores.std()

    print("\n=== Cross-Validation (on training split) ===")
    cv_report("LogReg", logreg, data)
    cv_report("DecisionTree", tree, data)
    cv_report("RandomForest", rf, data)

    # 4) Train on train split and evaluate on holdout test set
    print("\n=== Train & Evaluate on holdout test set ===")
    acc_lr, label_lr, cm_lr = train_and_eval(logreg, data, "Logistic Regression")
    acc_tree, label_tree, cm_tree = train_and_eval(tree, data, "Decision Tree")
    acc_rf, label_rf, cm_rf = train_and_eval(rf, data, "Random Forest")

    # plot confusion matrices (include accuracy in title if your plot function supports it)
    try:
        plot_cm(cm_lr, data.target_names, f"{label_lr} — Confusion Matrix", accuracy=acc_lr)
        plot_cm(cm_tree, data.target_names, f"{label_tree} — Confusion Matrix", accuracy=acc_tree)
        plot_cm(cm_rf, data.target_names, f"{label_rf} — Confusion Matrix", accuracy=acc_rf)
    except TypeError:
        # fallback if plot_cm doesn't accept accuracy param
        plot_cm(cm_lr, data.target_names, f"{label_lr} — Confusion Matrix")
        plot_cm(cm_tree, data.target_names, f"{label_tree} — Confusion Matrix")
        plot_cm(cm_rf, data.target_names, f"{label_rf} — Confusion Matrix")

    # 5) ROC curves (AUC) for the models that support predict_proba
    print("\n=== ROC / AUC ===")
    try:
        auc_lr = plot_roc(logreg, data.X_test, data.y_test, "Logistic Regression ROC")
        print(f"Logistic Regression AUC: {auc_lr:.3f}")
    except Exception as e:
        print("Could not plot ROC for Logistic Regression:", e)
        auc_lr = None

    try:
        auc_rf = plot_roc(rf, data.X_test, data.y_test, "Random Forest ROC")
        print(f"Random Forest AUC: {auc_rf:.3f}")
    except Exception as e:
        print("Could not plot ROC for Random Forest:", e)
        auc_rf = None

    # 6) Single canonical GridSearchCV block for Random Forest (small grid)
    print("\n=== Starting GridSearchCV for RandomForest (small grid) ===")
    param_grid = {
        "clf__n_estimators": [100, 300],      # small grid for reasonable runtime
        "clf__max_depth": [3, 5, None],
        "clf__min_samples_leaf": [1, 2],
    }

    grid = GridSearchCV(
        rf,                 # must be pipeline with final step named 'clf'
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(data.X_train, data.y_train)

    print("\nGridSearchCV complete.")
    print("Best parameters:", grid.best_params_)
    print("Best CV score (grid):", grid.best_score_)

    # Evaluate tuned RF on the holdout test set
    best_rf = grid.best_estimator_
    test_acc_best_rf = best_rf.score(data.X_test, data.y_test)
    try:
        test_auc_best_rf = plot_roc(best_rf, data.X_test, data.y_test, "Tuned RF ROC")
    except Exception:
        test_auc_best_rf = None

    print(f"Tuned RF Test Accuracy={test_acc_best_rf:.3f} | AUC={test_auc_best_rf:.3f}" if test_auc_best_rf is not None
          else f"Tuned RF Test Accuracy={test_acc_best_rf:.3f}")

    # Save the tuned model (only once)
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_rf, "models/best_rf_bcancer.joblib")
    print("✅ Saved tuned model to models/best_rf_bcancer.joblib")

    # Feature importances (tuned RF)
    rf_model = best_rf if 'best_rf' in globals() else rf
    rf_est = rf_model.named_steps['clf']
    importances = rf_est.feature_importances_
    fi = pd.Series(importances, index=data.feature_names).sort_values(ascending=False)
    print("\nTop 10 Random Forest feature importances:")
    print(fi.head(10))

    res = permutation_importance(best_rf, data.X_test, data.y_test, n_repeats=25, random_state=42, n_jobs=-1)
    perm = pd.Series(res.importances_mean, index=data.feature_names).sort_values(ascending=False)
    print("\nTop 10 Permutation importances (test):")
    print(perm.head(10))


    # Logistic Regression coefficients (after scaler)
    lr_est = logreg.named_steps['clf']
    coefs = lr_est.coef_.ravel()
    lr_coefs = pd.Series(coefs, index=data.feature_names).sort_values(key=np.abs, ascending=False)
    print("\nTop 10 Logistic Regression coefficients (by absolute value):")
    print(lr_coefs.head(10))

    # 7) Final summary
    print("\n=== Final Summary ===")
    print(f"Logistic Regression Acc: {acc_lr:.3f} | AUC: {auc_lr:.3f}" if auc_lr is not None else f"Logistic Regression Acc: {acc_lr:.3f}")
    print(f"Decision Tree Acc: {acc_tree:.3f}")
    print(f"Random Forest Acc: {acc_rf:.3f} | AUC: {auc_rf:.3f}" if auc_rf is not None else f"Random Forest Acc: {acc_rf:.3f}")
    print(f"Tuned Random Forest Test Acc: {test_acc_best_rf:.3f} | Tuned RF AUC: {test_auc_best_rf:.3f}" if test_auc_best_rf is not None else f"Tuned Random Forest Test Acc: {test_acc_best_rf:.3f}")

    # 8) Return objects optionally (useful for interactive debugging)
    return {
        "data": data,
        "logreg": logreg,
        "tree": tree,
        "rf": rf,
        "best_rf": best_rf,
        "metrics": {
            "acc_lr": acc_lr,
            "acc_tree": acc_tree,
            "acc_rf": acc_rf,
            "auc_lr": auc_lr,
            "auc_rf": auc_rf,
            "test_acc_best_rf": test_acc_best_rf,
            "test_auc_best_rf": test_auc_best_rf,
        }
    }

if __name__ == "__main__":
    main()

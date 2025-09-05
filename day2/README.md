# Day 2: Baseline Models on Breast Cancer Dataset 🎗️

This project continues the **self-paced journey to learn AI & machine learning programming**.  
The goal of Day 2 is to **reinforce the workflow from Day 1** but on a more realistic, binary classification dataset: the Breast Cancer Wisconsin dataset.

---

## 🎯 Learning Goals
- Practice the **same ML workflow** from Day 1 on a new dataset.  
- Understand **binary classification** problems and their evaluation.  
- Learn how to use **cross-validation (CV)** to get a more stable accuracy estimate.  
- Compare Logistic Regression, Decision Tree, and Random Forest on this dataset.  
- Learn why **false negatives vs false positives** matter differently depending on context.  

---

## 🗂️ Steps in the Script
1. **Load Dataset**  
   - Use the Breast Cancer Wisconsin dataset (569 tumor samples, 30 features).  
   - Target labels: *malignant* or *benign*.  
   - Split into training (80%) and testing (20%) with stratification.

2. **Quick EDA**  
   - Inspect feature names and shapes.  
   - Print summary statistics for the dataset.

3. **Build Models**  
   - Logistic Regression (with scaling).  
   - Decision Tree (with optional depth tuning).  
   - Random Forest (with tuned hyperparameters).  

4. **Cross-Validation (CV)**  
   - Run 5-fold CV on the training split for each model.  
   - Compare mean accuracy and stability (standard deviation).  
   - Understand that CV gives a better estimate than a single test split.

5. **Train and Evaluate**  
   - Fit each model on the training data.  
   - Predict on the holdout test set.  
   - Evaluate using:
     - Accuracy  
     - Classification report (precision, recall, F1)  
     - Confusion matrix (visualized, with accuracy in the title)  

6. **Compare Results**  
   - Print both CV means and holdout test accuracies.  
   - Inspect confusion matrices to understand **false negatives (FN)** vs **false positives (FP)**.  

---

## 📊 Why This Matters
- Binary classification is the most common real-world ML problem (spam detection, fraud detection, medical diagnosis, etc.).  
- In healthcare, **false negatives** (malignant → benign) are far more dangerous than false positives.  
- Learning to interpret both accuracy *and* confusion matrices is crucial for practical ML.  

---

## 🚀 Setup Instructions
This project uses [uv](https://github.com/astral-sh/uv) for fast Python environment management.

1. Create a virtual environment:
```bash
uv venv .venv
```

2. Activate it:  
- macOS/Linux:
```bash
source .venv/bin/activate
```
- Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

3. Install required packages:
```bash
uv pip install numpy pandas matplotlib scikit-learn jupyter
```

---

## ✅ Next Steps
- Experiment with **hyperparameters** (max_depth, n_estimators, min_samples_leaf).  
- Try additional models (Support Vector Machine, kNN).  
- Explore **normalized confusion matrices** to see per-class error rates.  
- Move on to **Week 1, Day 3**: deeper evaluation metrics (ROC curves, AUC).  

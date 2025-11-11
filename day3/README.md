# Day 3: Model Tuning & Hyperparameter Optimization on Breast Cancer Dataset 🎗️

This project continues the **self-paced journey to learn AI & machine learning programming**.  
The goal of Day 3 is to **advance beyond baseline models** by implementing hyperparameter tuning, GridSearchCV, and advanced model optimization techniques on the Breast Cancer Wisconsin dataset.

---

## 🎯 Learning Goals
- Build on **baseline models** with advanced tuning techniques
- Master **GridSearchCV** for systematic hyperparameter optimization
- Learn **threshold tuning** to balance false positives and false negatives
- Understand **ROC curves and AUC** for model comparison
- Practice **model persistence** (saving/loading trained models with joblib)
- Compare baseline vs tuned model performance  

---

## 🗂️ Steps in the Script
1. **Load Dataset**  
   - Use the Breast Cancer Wisconsin dataset (569 tumor samples, 30 features)
   - Target labels: *malignant* or *benign*
   - Split into training (80%) and testing (20%) with stratification

2. **Quick EDA**  
   - Inspect feature names and shapes
   - Print summary statistics for the dataset

3. **Build Baseline Models**  
   - Logistic Regression (with scaling)
   - Decision Tree (baseline configuration)
   - Random Forest (baseline with n_estimators=100)

4. **Cross-Validation (CV)**  
   - Run 5-fold CV on the training split for each model
   - Compare mean accuracy and stability (standard deviation)

5. **Train and Evaluate Baseline**  
   - Fit each model on the training data
   - Predict on the holdout test set
   - Evaluate using:
     - Accuracy  
     - Classification report (precision, recall, F1)  
     - Confusion matrix (visualized)

6. **ROC Curves & AUC**
   - Plot ROC curves for Logistic Regression and Random Forest
   - Calculate and compare AUC scores
   - Understand ROC/AUC as threshold-independent metrics

7. **GridSearchCV for Hyperparameter Tuning**
   - Define parameter grid for Random Forest:
     - `n_estimators`: [100, 300]
     - `max_depth`: [3, 5, None]
     - `min_samples_leaf`: [1, 2]
   - Run exhaustive grid search with 5-fold CV
   - Identify best hyperparameter combination
   - Evaluate tuned model on test set

8. **Model Persistence**
   - Save the best tuned model using `joblib`
   - Learn how to load and reuse trained models

9. **Compare Results**  
   - Compare baseline vs tuned model performance
   - Analyze whether tuning improved test accuracy and AUC  

---

## 📊 Why This Matters
- **Hyperparameter tuning** is essential for getting the best performance from ML models
- **GridSearchCV** automates the search process and prevents overfitting through CV
- Understanding **ROC/AUC** helps compare models independent of classification thresholds
- **Model persistence** enables deploying trained models in production
- In healthcare, **false negatives** (malignant → benign) are far more dangerous than false positives
- Learning to systematically optimize models is crucial for real-world ML applications  

---

## 🚀 Setup Instructions
This project uses a Python virtual environment for package management.

1. Create a virtual environment:
```bash
python3 -m venv venv
```

2. Activate it:  
- macOS/Linux:
```bash
source venv/bin/activate
```
- Windows (PowerShell):
```powershell
venv\Scripts\Activate.ps1
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy pandas matplotlib scikit-learn jupyter joblib
```

4. Run the script:
```bash
python day3_model_tuning.py
```

---

## 📦 Key Outputs
- **Confusion matrices** for all models (displayed as plots)
- **ROC curves** with AUC scores for Logistic Regression and Random Forest
- **GridSearchCV results** showing best hyperparameters
- **Saved model**: `models/best_rf_bcancer.joblib` (tuned Random Forest)
- **Feature importances** for Random Forest and Logistic Regression coefficients

---

## 📊 Results – Model Evaluation & Tuning

| Model | Accuracy | AUC | Notes |
|--------|-----------|-----|-------|
| Logistic Regression | 0.982 | 0.995 | Best generalization; interpretable |
| Decision Tree | 0.912 | — | Overfits; less stable |
| Random Forest (baseline) | 0.956 | 0.994 | Very strong baseline |
| Tuned Random Forest | 0.947 | 0.994 | GridSearchCV optimized; same AUC, slight accuracy trade-off |

**Top Features (shared across models):**  
`worst area`, `worst concave points`, `worst radius`, `mean concave points`

**Key Insights:**  
- Dataset is nearly linearly separable — hence Logistic Regression's strong performance
- Random Forest adds robustness to outliers but minimal accuracy gain on this dataset
- Tuning helped find optimal hyperparameters: `n_estimators=300`, `max_depth=None`, `min_samples_leaf=1`
- Feature redundancy present — many correlated features (mean/worst variants)
- Medical domain: False negatives are critical — consider threshold tuning for production

---

## ✅ Next Steps
- Experiment with **larger parameter grids** (more n_estimators, different max_depth values)
- Try **RandomizedSearchCV** for faster hyperparameter search on large grids
- Implement **threshold tuning** to optimize for specific metrics (minimize FN in healthcare)
- Explore **feature importance** from the Random Forest model
- Try additional models (Support Vector Machine, Gradient Boosting, XGBoost)
- Learn **ensemble methods** and stacking different models
- Move on to **Day 4**: Feature engineering and selection techniques  

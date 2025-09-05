# Day 1: Minimal Baseline on Iris Dataset 🌱

This project is part of a **self-paced journey to learn AI & machine learning programming**.  
The goal of Day 1 is to **build muscle memory for a full ML workflow**: from loading data → training models → evaluating results → visualizing errors.

---

## 🎯 Learning Goals
- Understand the **end-to-end ML workflow** on a simple dataset.  
- Learn how to use **scikit-learn pipelines** to combine preprocessing + models.  
- Compare a **linear model** (Logistic Regression) vs a **non-linear model** (Decision Tree).  
- Learn to read a **confusion matrix** and a **classification report**.  
- Get in the habit of doing a quick **Exploratory Data Analysis (EDA)**.  

---

## 🗂️ Steps in the Script
1. **Load Dataset**  
   - Use the classic Iris dataset (150 flower samples, 3 species).  
   - Split into training (80%) and testing (20%) with stratification.

2. **Quick EDA**  
   - Preview feature names and basic statistics.  
   - Build the habit of “looking at your data before modeling.”

3. **Build Models**  
   - Logistic Regression (with feature scaling).  
   - Decision Tree (no scaling needed).  
   - (Optional Stretch) Random Forest.

4. **Train and Evaluate**  
   - Train each model on training data.  
   - Predict on test data.  
   - Measure performance:
     - Accuracy  
     - Precision / Recall / F1 (via classification report)  
     - Confusion matrix (visualized)

5. **Compare Results**  
   - Print side-by-side model accuracies.  
   - Inspect confusion matrices to see *which species get confused*.  

6. **Baseline Winner**  
   - Declare the best-performing model as your baseline to beat in later days.  

---

## 📊 Why This Matters
- This script gives you a **baseline workflow** you’ll repeat throughout the 4-week program.  
- Future datasets will be larger and messier, but the process (EDA → models → metrics → confusion matrix) stays the same.  
- By practicing with Iris, you learn to **trust the workflow**, not just the final accuracy number.  

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
- Try the **stretch goals**: add Random Forest, cross-validation, or feature importance plots.  
- Move on to **Day 2 (Breast Cancer Dataset)** to practice on a more realistic, binary classification problem.  
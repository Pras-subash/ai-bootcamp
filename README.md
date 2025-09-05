# AI Bootcamp: Minimal ML Baselines

This repository contains hands-on machine learning exercises for beginners, focusing on building baseline models and understanding the end-to-end ML workflow.  
Each day introduces a new dataset and reinforces best practices in data analysis, modeling, and evaluation.

---

## 📅 Project Structure

- **day1/**: Baseline models on the Iris dataset (multi-class classification)
- **day2/**: Baseline models on the Breast Cancer Wisconsin dataset (binary classification)

Each folder includes:
- `dayX_*.py`: Main script for the day’s workflow
- `check_env.py`: Quick environment/version check
- `requirements.txt`: Python dependencies
- `README.md`: Day-specific instructions and learning goals

---

## 🚀 Getting Started

1. **Create a virtual environment**  
   (Recommended: [uv](https://github.com/astral-sh/uv) for fast setup)
   ```bash
   uv venv .venv
   ```

2. **Activate the environment**
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     .venv\Scripts\Activate.ps1
     ```

3. **Install dependencies**
   ```bash
   uv pip install numpy pandas matplotlib scikit-learn jupyter
   ```

4. **Run environment check**
   ```bash
   python day1/check_env.py
   python day2/check_env.py
   ```

5. **Run baseline scripts**
   ```bash
   python day1/day1_iris_baseline.py
   python day2/day2_bcancer_baseline.py
   ```

---

## 📊 Learning Goals

- Practice the full ML workflow: data loading, EDA, modeling, evaluation, and visualization.
- Compare linear and tree-based models (Logistic Regression, Decision Tree, Random Forest).
- Understand confusion matrices and classification reports.
- Learn cross-validation for robust accuracy estimation.
- Explore the impact of false positives/negatives in real-world contexts.

---

## ✅ Next Steps

- Experiment with hyperparameters and additional models.
- Try normalized confusion matrices and ROC/AUC metrics.
- Continue to future days for deeper ML concepts.

---

For details on each day’s see [day1/README.md](day1/README.md) and [day2/README.md](day2/README.md).
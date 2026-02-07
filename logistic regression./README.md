# Logistic Regression on Iris Dataset 🌸📊

This project demonstrates **Logistic Regression** using **scikit-learn** to perform **binary classification** on the Iris dataset.  
The model predicts whether a flower belongs to the **Iris Virginica** species based solely on **petal width**.

---

## Dataset

- **Source:** `sklearn.datasets.load_iris`
- **Samples:** 150
- **Feature Used:**  
  - Petal width (cm)
- **Target Encoding:**
  - `1` → Iris Virginica
  - `0` → Not Virginica

---

## Model Details

- **Algorithm:** Logistic Regression
- **Problem Type:** Binary Classification
- **Feature Dimension:** 1 (univariate)
- **Solver:** Default scikit-learn solver

---

## Requirements

- Python 3.7+
- NumPy
- scikit-learn
- matplotlib

Install dependencies:

```bash
pip install numpy scikit-learn matplotlib

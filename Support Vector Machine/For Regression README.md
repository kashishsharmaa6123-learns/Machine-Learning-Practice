# Support Vector Regression (SVR) on California Housing Dataset 🏠📈

This project demonstrates **Support Vector Regression (SVR)** using an **RBF kernel** on the **California Housing dataset** from scikit-learn.  
The model predicts house values based on a single feature: **average number of rooms per household**.

---

## Dataset

- **Source:** `sklearn.datasets.fetch_california_housing`
- **Samples:** 20,640
- **Features Used:**  
  - `AveRooms` (average rooms per household)
- **Target:**  
  - Median house value (in hundreds of thousands of dollars)

---

## Model Details

- **Algorithm:** Support Vector Regression (SVR)
- **Kernel:** Radial Basis Function (RBF)
- **Hyperparameters:**
  - `C = 100`
  - `gamma = 0.1`
  - `epsilon = 0.1`
- **Train/Test Split:** 70% / 30%
- **Random State:** 42

---

## Requirements

- Python 3.8+
- scikit-learn
- matplotlib

Install dependencies:

```bash
pip install scikit-learn matplotlib

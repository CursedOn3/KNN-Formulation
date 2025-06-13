# k-NN Classifier with Cross-Validation and Visualization

This project implements a custom **k-Nearest Neighbors (k-NN)** classifier using Euclidean distance from scratch, visualizes training data and test points, and performs **Stratified K-Fold Cross-Validation** to determine the best `k`.

---

## 🔧 Features

- ✅ Custom k-NN prediction function
- 📈 Data visualization with `matplotlib` and `seaborn`
- 🤖 Optimal `k` selection via cross-validation
- 🔁 Dynamic `k` generation based on dataset size
- 🧠 Smart suggestion of `n_splits` for StratifiedKFold

---

## 📌 How It Works
A suitable k is calculated from the number of training samples (√n, made odd).

A test point is classified using the custom knn_predict function.

The dataset is plotted for visual analysis.

Cross-validation is used to find the best k using StratifiedKFold.


## 📈 Visual Output
Scatter plot of training samples categorized by label.

Highlighted test point (X) on the same plot.

Plot of cross-validated accuracy vs. different k values.


## 🧪 Dependencies
Install with pip: pip install numpy pandas seaborn matplotlib scikit-learn



## ▶️ Run the Script
python main.py



## 📊 Dataset

- Each data point includes:
  - `Age` (19–45)
  - `Salary` (₹15,000–30,000)
- Labeled into 3 categories: `A`, `B`, and `C`

> Example:
```python
training_data = [
    [22, 24000], [26, 25500], ...
]
training_labels = ['A', 'A', 'B', ...]




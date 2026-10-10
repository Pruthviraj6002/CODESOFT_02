# Task 2 — Credit Card Fraud Detection

A machine-learning project that classifies credit-card transactions as **fraudulent (1)** or **legitimate (0)**. It compares Logistic Regression, Decision Tree, and Random Forest models, with special attention to class imbalance.

## Project structure

```text
credit_card_fraud_detection_task2/
├── data/
│   └── README.md
├── models/                 # Saved fitted models (created after training)
├── notebooks/
│   └── task2_credit_card_fraud_detection.ipynb
├── results/                # Metrics, confusion matrices, and plots (created after training)
├── src/
│   └── train.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 1. Dataset

Use the public **Credit Card Fraud Detection** dataset from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download `creditcard.csv` and place it at:

```text
data/creditcard.csv
```

The dataset has anonymized transaction features (`V1`–`V28`), `Time`, `Amount`, and a target column named `Class` (`1` = fraud, `0` = legitimate). It is highly imbalanced: fraudulent transactions are rare. Do not commit the dataset to GitHub unless you have checked its license and repository-size constraints.

## 2. Setup

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
```

Activate it:

- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Run the project

From the project root:

```bash
python src/train.py --data data/creditcard.csv
```

Optional arguments:

```bash
python src/train.py --data data/creditcard.csv --test-size 0.2 --random-state 42
```

The script will:
- validate the input dataset and target column;
- make a stratified train/test split;
- train Logistic Regression, Decision Tree, and Random Forest;
- use class weights to account for class imbalance;
- evaluate models on the untouched test set;
- save a metrics CSV, confusion-matrix plots, a precision-recall curve, and fitted model files.

Outputs are written to `results/` and `models/`.

## 4. Evaluation

Accuracy alone is misleading for this dataset because the majority class dominates. Review:

- **Precision:** among transactions flagged as fraud, how many were fraud?
- **Recall:** how many actual fraud transactions were caught?
- **F1-score:** balance between precision and recall.
- **PR-AUC / Average Precision:** useful when the positive class is rare.
- **ROC-AUC:** ranking quality across thresholds.
- **Confusion matrix:** true/false positives and negatives.

The default classification threshold is 0.5. In a real fraud system, choose the threshold based on fraud-loss cost, false-decline cost, and investigation capacity. Results from this public dataset are educational and are not evidence of production readiness.

## 5. Notebook

Open `notebooks/task2_credit_card_fraud_detection.ipynb` in Jupyter or Google Colab. The notebook explains the workflow and calls the same training script so results are consistent.

## Limitations and responsible use

- The data is anonymized and historical; patterns may not reflect current fraud.
- Do not use this model alone to block real transactions.
- Avoid data leakage: preprocessing and training happen after the split, and evaluation uses held-out test data.
- Fraud detection is a threshold and cost-sensitive problem, not simply an accuracy contest.

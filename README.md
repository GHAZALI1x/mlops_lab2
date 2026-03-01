# MLOps Lab 2 — DVC Pipeline

## Overview
DVC-based ML pipeline on the Iris classification dataset.
Three experiments (branches): Logistic Regression · Random Forest · SVC.

## Dataset
[Iris CSV](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv) — 150 samples, 4 features, 3 classes.

## Pipeline Stages
| Stage | Script | Input | Output |
|---|---|---|---|
| get_data | `get_data.py` | URL | `data/data_raw.csv` |
| preprocess | `preprocess.py` | `data_raw.csv` | `train.csv`, `test.csv` |
| train | `train.py` | `train.csv` | `models/model.pkl` |
| validate | `validate.py` | `model.pkl`, `test.csv` | `metrics.json`, `confusion_matrix.png` |

## Branches & Models
| Branch | Model |
|---|---|
| `main` | LogisticRegression |
| `random-forest` | RandomForestClassifier |
| `svc-model` | SVC |

## Setup
```bash
pip install dvc scikit-learn pandas joblib matplotlib seaborn
dvc init
dvc remote add mylocal ../dvc_remote
dvc remote default mylocal
```

## Run Pipeline
```bash
dvc repro --force
```

## Compare Experiments
```bash
# Numerical metrics
dvc metrics show --all-branches

# Visual comparison
dvc plots diff main random-forest svc-model
```

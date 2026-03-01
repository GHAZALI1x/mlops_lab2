import pandas as pd
import joblib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import os

# Load test data
test_df = pd.read_csv('data/test.csv')
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

# Load trained model
model = joblib.load('models/model.pkl')

# Predict
preds = model.predict(X_test)

# --- Metrics ---
acc       = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average='weighted', zero_division=0)
recall    = recall_score(y_test, preds, average='weighted', zero_division=0)
f1        = f1_score(y_test, preds, average='weighted', zero_division=0)

metrics = {
    "accuracy":  round(acc, 5),
    "precision": round(precision, 5),
    "recall":    round(recall, 5),
    "f1_score":  round(f1, 5)
}

# Save metrics.json
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v}")

# --- Confusion Matrix Plot ---
cm = confusion_matrix(y_test, preds, labels=sorted(y_test.unique()))
class_names = ['setosa', 'versicolor', 'virginica']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names,
            cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
model_name = type(model).__name__
plt.title(f'Confusion Matrix — {model_name}')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()

print(f"Confusion matrix saved → confusion_matrix.png")

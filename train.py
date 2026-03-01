import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load training data
train_df = pd.read_csv('data/train.csv')
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

# Train model - Logistic Regression (main branch)
model = LogisticRegression(random_state=42, max_iter=200)
model.fit(X_train, y_train)

# Save trained model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')

print(f"Model trained: LogisticRegression")
print(f"Training samples: {len(X_train)}")
print(f"Model saved → models/model.pkl")

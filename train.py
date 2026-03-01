import pandas as pd
from sklearn.svm import SVC
import joblib
import os

train_df = pd.read_csv('data/train.csv')
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')

print(f"Model trained: SVC (kernel=rbf)")
print(f"Training samples: {len(X_train)}")
print(f"Model saved → models/model.pkl")
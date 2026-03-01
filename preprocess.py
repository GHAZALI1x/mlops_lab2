import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Load raw data
df = pd.read_csv('data/data_raw.csv')
print(f"Loaded raw data: {df.shape[0]} rows")

# --- Clean ---
# Drop any duplicate rows
df = df.drop_duplicates()

# Drop rows with missing values
df = df.dropna()

# Encode target label (species) to integers
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

print(f"Classes: {list(le.classes_)}")
print(f"After cleaning: {df.shape[0]} rows")

# --- Split ---
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['species'])

# Ensure output directory exists
os.makedirs('data', exist_ok=True)

# Save splits
train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

print(f"Train set: {len(train_df)} rows  →  data/train.csv")
print(f"Test  set: {len(test_df)} rows  →  data/test.csv")

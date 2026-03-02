import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Ensure directories exist
os.makedirs('data', exist_ok=True)

# Load iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# First split: train+val and test (20% for test)
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Second split: train and val (25% of train_val is 20% of total)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)

# Save to CSV
train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

print(f'Data prepared: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test.')

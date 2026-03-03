import os
import pandas as pd
import numpy as np

# Ensure directories exist
os.makedirs('data', exist_ok=True)

# Load data from Parquet
data_path = 'data/data.parquet'
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found.")
    exit(1)

print(f"Reading data from {data_path}")
df = pd.read_parquet(data_path)

# Data specific configuration based on schema inspection
time_col = 'as_of_date'
target_col = 'y'
categorical_cols = ['Weather Condition', 'Seasonality']

# Feature Engineering for Time Series
df[time_col] = pd.to_datetime(df[time_col])
df = df.sort_values(time_col)

# Convert categorical objects to category type for LightGBM
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Drop rows with NaN if any (especially from lags already in the data)
df = df.dropna(subset=[target_col])

# Chronological split
total_len = len(df)
train_end = int(total_len * 0.7)
val_end = int(total_len * 0.85)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

# Save to CSV for the rest of the pipeline
# Note: categories will be lost in CSV, so we might want to use Joblib/Parquet 
# but for now we'll handle re-categorization in train_model.py if needed.
train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

print(f'Data prepared based on specific schema: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test.')

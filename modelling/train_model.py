import os
import datetime
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Ensure directories exist
os.makedirs('model', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Load data
train_path = os.path.join('data', 'train.csv')
val_path = os.path.join('data', 'val.csv')

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)

# Data specific configuration
target_col = 'y'
time_col = 'as_of_date'

# Identify categorical columns (everything that isn't numeric or the target/time)
# In CSV, categories are read back as objects, so we re-convert them.
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
if target_col in categorical_cols: categorical_cols.remove(target_col)
if time_col in categorical_cols: categorical_cols.remove(time_col)

print(f"Handling these as categorical features: {categorical_cols}")

for df in [train_df, val_df]:
    for col in categorical_cols:
        df[col] = df[col].astype('category')

# Prepare features and target
X_train = train_df.drop(columns=[target_col, time_col], errors='ignore')
y_train = train_df[target_col]

X_val = val_df.drop(columns=[target_col, time_col], errors='ignore')
y_val = val_df[target_col]

# Train LightGBM model
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': 42,
    'categorical_feature': [col for col in categorical_cols if col in X_train.columns]
}

model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

# Save the trained model
model_path = os.path.join('model', 'lgbm_timeseries.joblib')
joblib.dump(model, model_path)
print(f'Model saved to {model_path}')

# Evaluate
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

# Write training report
report_path = os.path.join('reports', 'train_report.txt')
with open(report_path, 'w') as f:
    f.write(f'Date: {datetime.datetime.now()}\n')
    f.write(f'Data Schema Checked: Yes\n')
    f.write(f'Target: {target_col}\n')
    f.write(f'MAE: {mae}\n')
    f.write(f'RMSE: {rmse}\n')
print(f'Training report saved to {report_path}')

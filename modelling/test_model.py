import os
import datetime
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Paths
model_path = os.path.join('model', 'lgbm_timeseries.joblib')
test_path = os.path.join('data', 'test.csv')

# Load model
if not os.path.isfile(model_path):
    print(f"Model not found at {model_path}.")
    exit(1)
model = joblib.load(model_path)

# Load test dataset
if not os.path.isfile(test_path):
    print(f"Test data not found at {test_path}.")
    exit(1)
df = pd.read_csv(test_path)

# Data specific configuration
target_col = 'y'
time_col = 'as_of_date'
categorical_cols = ['Weather Condition', 'Seasonality']

# Re-convert categorical columns
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

X_test = df.drop(columns=[target_col, time_col], errors='ignore')
y_test = df[target_col]

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Write test report
os.makedirs('reports', exist_ok=True)
report_path = os.path.join('reports', 'test_report.txt')
with open(report_path, 'w') as f:
    f.write(f'Date: {datetime.datetime.now()}\n')
    f.write(f'MAE: {mae}\n')
    f.write(f'RMSE: {rmse}\n')

print(f'Test report saved to {report_path}')

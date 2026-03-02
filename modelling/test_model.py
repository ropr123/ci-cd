import os
import datetime
import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Paths
# Paths
model_path = os.path.join('model', 'rf_iris.joblib')
test_path = os.path.join('data', 'test.csv')

# Load model
if not os.path.isfile(model_path):
    print(f"Model not found at {model_path}. Please run train_model.py first.")
    exit(1)
clf = joblib.load(model_path)

# Load test dataset
if not os.path.isfile(test_path):
    print(f"Test data not found at {test_path}. Please run data_prep.py first.")
    exit(1)
df = pd.read_csv(test_path)
X_test = df.drop(columns='target')
y_test = df['target']

# Predict and evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Write test report
os.makedirs('reports', exist_ok=True)
report_path = os.path.join('reports', 'test_report.txt')
with open(report_path, 'w') as f:
    f.write(f'Date: {datetime.datetime.now()}\n')
    f.write(f'Accuracy: {acc}\n')
    f.write(report)

print(f'Test report saved to {report_path}')

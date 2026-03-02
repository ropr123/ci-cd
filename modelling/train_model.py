import os
import datetime
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('model', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Load data from CSV
train_path = os.path.join('data', 'train.csv')
df = pd.read_csv(train_path)
X_train = df.drop(columns='target')
y_train = df['target']

# Train classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Save the trained model
model_path = os.path.join('model', 'rf_iris.joblib')
joblib.dump(clf, model_path)
print(f'Model saved to {model_path}')

# Load validation data for internal evaluation
val_path = os.path.join('data', 'val.csv')
val_df = pd.read_csv(val_path)
X_val = val_df.drop(columns='target')
y_val = val_df['target']

# Evaluate on validation set
y_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred)

# Write training report
report_path = os.path.join('reports', 'train_report.txt')
with open(report_path, 'w') as f:
    f.write(f'Date: {datetime.datetime.now()}\n')
    f.write(f'Accuracy: {acc}\n')
    f.write(report)
print(f'Training report saved to {report_path}')

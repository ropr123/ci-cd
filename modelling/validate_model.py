import os
import datetime
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Paths
model_path = os.path.join('model', 'rf_iris.joblib')
val_path = os.path.join('data', 'val.csv')

# Load model
if not os.path.isfile(model_path):
    print(f"Model not found at {model_path}. Please run train_model.py first.")
    exit(1)
clf = joblib.load(model_path)

# Load validation data
df = pd.read_csv(val_path)
X = df.drop(columns='target')
y = df['target']

# Perform 5-fold cross-validation on the validation set (or just evaluate)
# Here we'll just evaluate the pre-trained model on the validation set for simplicity,
# or we can do CV if the user specifically wanted that for model selection.
# Given the user's script used CV, I'll keep it but typically CV is on training data.
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
mean_acc = scores.mean()

# Write validation report
report_path = os.path.join('reports', 'validation_report.txt')
with open(report_path, 'w') as f:
    f.write(f'Date: {datetime.datetime.now()}\n')
    f.write('Cross-validation accuracies on validation set:\n')
    for i, s in enumerate(scores, 1):
        f.write(f'Fold {i}: {s}\n')
    f.write(f'Average accuracy: {mean_acc}\n')

print(f'Validation report saved to {report_path}')

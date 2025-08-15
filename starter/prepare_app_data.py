import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load models and test data
logreg = joblib.load("best_model_logreg.pkl")
rf = joblib.load("best_model_random_forest.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

# Predict
y_pred_logreg = logreg.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Function to get metrics
def get_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

# Save predictions and metrics
results = {
    "Logistic Regression": get_metrics(y_test, y_pred_logreg),
    "Random Forest": get_metrics(y_test, y_pred_rf)
}

joblib.dump(results, "model_metrics.pkl")
joblib.dump({"logreg": y_pred_logreg, "rf": y_pred_rf}, "y_preds.pkl")

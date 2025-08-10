from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def test_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    assert 0 <= acc <= 1, "Accuracy out of bounds"
    assert '0' in report and '1' in report, "Classification report missing classes"
    
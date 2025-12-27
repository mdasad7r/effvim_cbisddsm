import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_binary_metrics_from_logits(y_true: np.ndarray, logits: np.ndarray, thr: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probs = sigmoid(np.asarray(logits))
    y_pred = (probs >= thr).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float("nan")
    auc = float("nan")

    if len(np.unique(y_true)) == 2:
        f1 = float(f1_score(y_true, y_pred))
        auc = float(roc_auc_score(y_true, probs))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sens = float(tp / (tp + fn)) if (tp + fn) else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) else 0.0

    return {"acc": acc, "auc": auc, "sens": sens, "spec": spec, "f1": f1}

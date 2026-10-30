"""
Run ONCE to train the models, evaluate them, and save everything the app needs.

    python train_models.py

Reads : data/Disease_raw_v2.csv   (symptom columns + 'Disease' target)
Writes: models/*.pkl              (5 base models + stacking_model + imputer)
        models/metadata.json      (feature names/order, class labels)
        reports/metrics.json      (REAL evaluation results used by the app)

If you already have your own trained models, you do not need this script:
put your .pkl files in models/ using the same file names and edit
models/metadata.json (see README).
"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             classification_report)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).parent
DATA_FILE = BASE / "data" / "Disease_raw_v2.csv"
TARGET = "Disease"
RANDOM_STATE = 42
TEST_SIZE = 0.2
# Be honest about the data source - this text is shown on the About page.
DATASET_NOTE = ("Synthetic (simulated) symptom dataset generated from the symptom "
                "patterns of Disease.csv. It is not real patient data.")


def expected_calibration_error(y_true, proba, classes, n_bins=10):
    """ECE of the top-label confidence (lower = better calibrated)."""
    pred_idx = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    correct = (np.array(classes)[pred_idx] == np.array(y_true)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def main():
    (BASE / "models").mkdir(exist_ok=True)
    (BASE / "reports").mkdir(exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].str.strip()
    features = [c for c in df.columns if c != TARGET]

    # Split first -> imputer never sees test data
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                   stratify=df[TARGET])
    imputer = SimpleImputer(strategy="most_frequent")
    X_train = pd.DataFrame(imputer.fit_transform(train[features]), columns=features).astype(int)
    X_test = pd.DataFrame(imputer.transform(test[features]), columns=features).astype(int)
    y_train, y_test = train[TARGET].values, test[TARGET].values

    base = [
        ("dt", DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ("svm", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ("nb", BernoulliNB()),
        ("knn", KNeighborsClassifier(n_neighbors=7)),
    ]
    models = {
        "decision_tree": base[0][1], "random_forest": base[1][1], "svm": base[2][1],
        "naive_bayes": base[3][1], "knn": base[4][1],
        "stacking_model": StackingClassifier(
            estimators=base, final_estimator=LogisticRegression(max_iter=1000), cv=5),
    }

    classes = None
    metrics = {}
    for key, model in models.items():
        model.fit(X_train, y_train)
        classes = list(model.classes_)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        y_bin = label_binarize(y_test, classes=classes)
        metrics[key] = {
            "accuracy": accuracy_score(y_test, pred) * 100,
            "precision": precision_score(y_test, pred, average="weighted", zero_division=0) * 100,
            "recall": recall_score(y_test, pred, average="weighted", zero_division=0) * 100,
            "f1": f1_score(y_test, pred, average="weighted", zero_division=0) * 100,
            "roc_auc": roc_auc_score(y_bin, proba, average="weighted",
                                     multi_class="ovr") * 100,
        }
        metrics[key] = {k: round(v, 2) for k, v in metrics[key].items()}
        joblib.dump(model, BASE / "models" / f"{key}.pkl")
        print(f"{key:15s} accuracy = {metrics[key]['accuracy']:.2f}%")

    # Details for the final (stacking) model
    stack = models["stacking_model"]
    pred = stack.predict(X_test)
    proba = stack.predict_proba(X_test)
    ece = expected_calibration_error(y_test, proba, classes)
    rep = classification_report(y_test, pred, output_dict=True, zero_division=0)
    per_class = {c: {"precision": round(rep[c]["precision"] * 100, 2),
                     "recall": round(rep[c]["recall"] * 100, 2),
                     "f1": round(rep[c]["f1-score"] * 100, 2),
                     "support": int(rep[c]["support"])} for c in classes}

    joblib.dump(imputer, BASE / "models" / "imputer.pkl")
    (BASE / "models" / "metadata.json").write_text(json.dumps({
        "features": features,
        "classes": classes,
        "target": TARGET,
        "primary_model": "stacking_model",
        "imputer_strategy": "most_frequent",
        "dataset_note": DATASET_NOTE,
    }, indent=2))

    (BASE / "reports" / "metrics.json").write_text(json.dumps({
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": {"file": DATA_FILE.name, "n_rows": int(len(df)),
                    "n_train": int(len(train)), "n_test": int(len(test)),
                    "test_size": TEST_SIZE, "random_state": RANDOM_STATE},
        "models": metrics,
        "stacking": {
            "ece_percent": round(ece * 100, 2),
            # Probabilities are only shown in the app when measured calibration
            # error on the held-out test set is small (<= 10 %).
            "probabilities_reliable": bool(ece <= 0.10),
            "per_class": per_class,
            "confusion_matrix": confusion_matrix(y_test, pred, labels=classes).tolist(),
        },
    }, indent=2))
    print(f"\nStacking calibration error (ECE): {ece * 100:.2f}%")
    print("Saved models/, models/metadata.json, reports/metrics.json")


if __name__ == "__main__":
    main()

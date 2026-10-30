"""Loads the trained models and runs predictions. Nothing here is retrained or faked."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib

import preprocessing

BASE = Path(__file__).parent
MODELS_DIR = BASE / "models"
METRICS_FILE = BASE / "reports" / "metrics.json"

MODEL_FILES = {
    "stacking_model": "stacking_model.pkl",
    "decision_tree": "decision_tree.pkl",
    "random_forest": "random_forest.pkl",
    "svm": "svm.pkl",
    "knn": "knn.pkl",
    "naive_bayes": "naive_bayes.pkl",
}
MODEL_LABELS = {
    "stacking_model": "Stacking Ensemble",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svm": "SVM",
    "knn": "KNN",
    "naive_bayes": "Naive Bayes",
}
FALLBACK_ORDER = ["random_forest", "svm", "naive_bayes", "knn", "decision_tree"]


class PredictionError(Exception):
    pass


@dataclass
class Artifacts:
    models: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    imputer: object = None
    metrics: dict = None
    errors: list = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return bool(self.models) and bool(self.meta)

    @property
    def primary_key(self):
        wanted = self.meta.get("primary_model", "stacking_model")
        if wanted in self.models:
            return wanted
        for k in FALLBACK_ORDER:
            if k in self.models:
                return k
        return None


def load_artifacts() -> Artifacts:
    art = Artifacts()
    try:
        art.meta = preprocessing.load_metadata()
    except Exception as e:
        art.errors.append(f"Could not read models/metadata.json ({e}). Run train_models.py first.")
        return art
    art.imputer = preprocessing.load_imputer()
    for key, fname in MODEL_FILES.items():
        path = MODELS_DIR / fname
        if not path.exists():
            art.errors.append(f"Model file not found: models/{fname}")
            continue
        try:
            art.models[key] = joblib.load(path)
        except Exception as e:
            art.errors.append(f"Could not load models/{fname}: {e}")
    if METRICS_FILE.exists():
        try:
            art.metrics = json.loads(METRICS_FILE.read_text())
        except Exception as e:
            art.errors.append(f"Could not read reports/metrics.json: {e}")
    return art


def run_prediction(art: Artifacts, selected_symptoms, compare=True) -> dict:
    if not art.ready:
        raise PredictionError("Models are not loaded.")
    features = art.meta["features"]
    try:
        X = preprocessing.build_input_frame(selected_symptoms, features)
        X = preprocessing.preprocess(X, features, art.imputer)
    except Exception as e:
        raise PredictionError(f"Input preprocessing failed: {e}")

    key = art.primary_key
    model = art.models[key]
    try:
        label = str(model.predict(X)[0])
    except Exception as e:
        raise PredictionError(f"Prediction failed with {MODEL_LABELS.get(key, key)}: {e}")

    result = {
        "model_used": MODEL_LABELS.get(key, key),
        "predicted_disease": label,
        "primary_is_fallback": key != art.meta.get("primary_model", "stacking_model"),
        "individual": {},
        "scores": None,
    }

    if compare:
        for k, m in art.models.items():
            if k == key:
                continue
            try:
                result["individual"][MODEL_LABELS.get(k, k)] = str(m.predict(X)[0])
            except Exception:
                result["individual"][MODEL_LABELS.get(k, k)] = "unavailable"

    # Probabilities: only when the model supports them AND the measured calibration
    # error on the held-out test set (saved by train_models.py) is small.
    reliable = bool(art.metrics and art.metrics.get("stacking", {}).get("probabilities_reliable"))
    if key == "stacking_model" and reliable and hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[0]
            pairs = sorted(zip(model.classes_, proba), key=lambda t: t[1], reverse=True)
            result["scores"] = [(str(c), round(float(p) * 100, 1)) for c, p in pairs[:5]]
            result["ece"] = art.metrics["stacking"].get("ece_percent")
        except Exception:
            result["scores"] = None
    return result

"""Same preprocessing as the training pipeline + input validation."""
import json
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).parent / "models"


def pretty(feature: str) -> str:
    return feature.replace("_", " ")


def load_metadata() -> dict:
    return json.loads((MODELS_DIR / "metadata.json").read_text())


def load_imputer():
    path = MODELS_DIR / "imputer.pkl"
    return joblib.load(path) if path.exists() else None


def build_input_frame(selected_symptoms, features) -> pd.DataFrame:
    """One row, columns in the exact training order; 1 = symptom present."""
    unknown = set(selected_symptoms) - set(features)
    if unknown:
        raise ValueError(f"Unknown symptom(s): {sorted(unknown)}")
    return pd.DataFrame([[1 if f in selected_symptoms else 0 for f in features]],
                        columns=list(features))


def preprocess(df: pd.DataFrame, features, imputer=None) -> pd.DataFrame:
    """Order columns like training, then apply the saved imputer (missing values)."""
    missing_cols = [f for f in features if f not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature(s): {missing_cols}")
    df = df[list(features)]
    if imputer is not None:
        df = pd.DataFrame(imputer.transform(df), columns=list(features))
    return df.astype(int)


def calc_bmi(weight_kg, height_cm):
    if weight_kg and height_cm and height_cm > 0:
        return round(weight_kg / ((height_cm / 100) ** 2), 1)
    return None


def validate_patient(ref, age, weight_kg, height_cm, n_symptoms) -> list:
    errors = []
    if not ref or not ref.strip():
        errors.append("Please enter a patient name or ID.")
    elif len(ref.strip()) > 60:
        errors.append("Patient name/ID must be 60 characters or fewer.")
    if age is None or not (1 <= age <= 120):
        errors.append("Age must be between 1 and 120 years.")
    if weight_kg is None or not (2 <= weight_kg <= 300):
        errors.append("Weight must be between 2 and 300 kg.")
    if height_cm and not (30 <= height_cm <= 250):
        errors.append("Height must be between 30 and 250 cm (or leave it as 0 to skip).")
    if n_symptoms == 0:
        errors.append("Please select at least one symptom.")
    return errors

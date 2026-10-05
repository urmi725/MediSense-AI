"""
predict.py
-----------
The core prediction pipeline used by app.py:

  1. encode_symptoms()      -> multi-hot vector in the order the model expects
  2. predict_disease()      -> top-3 diseases + confidence % from the ML model
  3. calc_risk()             -> Low / Medium / High risk score
  4. detect_emergency()      -> True/False + which rule triggered
  5. get_recommendations()  -> precautions, medication info, specialist
  6. run_pipeline()          -> ties it all together for a single patient
"""

import os
import json
import csv
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

_model = None
_symptom_columns = None
_label_encoder = None


def _load_model():
    """Lazy-load model artifacts once and cache them at module level."""
    global _model, _symptom_columns, _label_encoder
    if _model is None:
        _model = joblib.load(os.path.join(MODEL_DIR, "disease_model.pkl"))
        _symptom_columns = joblib.load(os.path.join(MODEL_DIR, "symptom_encoder.pkl"))
        _label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    return _model, _symptom_columns, _label_encoder


def _load_csv_dict(filename, key_col, value_col, split_pipe=False):
    path = os.path.join(DATASET_DIR, filename)
    result = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = row[value_col]
            if split_pipe:
                value = [v.strip() for v in value.split("|")]
            result[row[key_col]] = value
    return result


def load_symptom_list():
    """Returns list of (symptom_key, display_name, severity_weight) for the UI."""
    path = os.path.join(DATASET_DIR, "symptoms.csv")
    symptoms = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symptoms.append(
                (row["symptom"], row["display_name"], int(row["severity_weight"]))
            )
    return sorted(symptoms, key=lambda s: s[1])


def load_emergency_rules():
    """Symptom sets that should trigger an immediate emergency banner."""
    # Kept in sync with dataset/disease_symptom_profiles.py at generation time;
    # duplicated here (as plain data) so predict.py has no import-time
    # dependency on the generator script.
    return [
        {"chest_pain", "shortness_of_breath"},
        {"chest_pain", "rapid_heartbeat", "sweating"},
        {"high_fever", "rapid_breathing"},
        {"shortness_of_breath", "blurred_vision"},
        {"vomiting", "high_fever", "abdominal_pain"},
    ]


def encode_symptoms(selected_symptoms):
    """
    Convert a list of symptom keys (e.g. ['cough', 'high_fever']) into the
    multi-hot feature vector the model expects, as a single-row DataFrame.
    """
    _, symptom_columns, _ = _load_model()
    selected_set = set(selected_symptoms)
    row = {col: (1 if col in selected_set else 0) for col in symptom_columns}
    return pd.DataFrame([row], columns=symptom_columns)


def predict_disease(selected_symptoms, top_n=3):
    """
    Returns a list of (disease_name, confidence_percent) tuples, sorted by
    confidence descending, for the top_n most likely diseases.
    """
    model, symptom_columns, label_encoder = _load_model()
    X = encode_symptoms(selected_symptoms)

    if X[symptom_columns].sum(axis=1).iloc[0] == 0:
        return []

    probabilities = model.predict_proba(X)[0]
    disease_names = label_encoder.classes_

    ranked = sorted(zip(disease_names, probabilities), key=lambda x: x[1], reverse=True)
    top = ranked[:top_n]
    return [(name, round(prob * 100, 1)) for name, prob in top]


def bmi_category(bmi):
    """Standard WHO BMI categories (informational only)."""
    if bmi is None:
        return None
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def load_model_metrics():
    """Reads model/metrics.json (written by train_model.py), if present."""
    path = os.path.join(MODEL_DIR, "metrics.json")
    if not os.path.exists(path):
        return None
    import json
    with open(path) as f:
        return json.load(f)


def calc_severity_score(selected_symptoms):
    """Sum of severity weights for the selected symptoms (from symptoms.csv)."""
    symptom_meta = {s[0]: s[2] for s in load_symptom_list()}
    return sum(symptom_meta.get(s, 1) for s in selected_symptoms)


def calc_risk(age, temperature_c, existing_conditions, selected_symptoms):
    """
    Combines age, fever, breathing-related symptoms, existing conditions, and
    overall symptom severity into a simple Low / Medium / High risk score.

    This is a rule-based heuristic for demo purposes — NOT a validated
    clinical risk model.
    """
    score = 0
    selected_set = set(selected_symptoms)

    # Age factor
    if age is not None:
        if age >= 65 or age <= 5:
            score += 2
        elif age >= 50:
            score += 1

    # Fever factor
    if temperature_c is not None:
        if temperature_c >= 39.0:
            score += 3
        elif temperature_c >= 38.0:
            score += 1

    # Breathing-related symptoms are weighted heavily
    breathing_symptoms = {"shortness_of_breath", "wheezing", "chest_tightness", "rapid_breathing"}
    score += 2 * len(selected_set & breathing_symptoms)

    # Existing conditions raise baseline risk
    if existing_conditions:
        score += min(len(existing_conditions), 3)

    # General symptom severity
    severity = calc_severity_score(selected_symptoms)
    score += severity // 3  # scale down since severity can add up fast

    if score >= 8:
        level = "High"
    elif score >= 4:
        level = "Medium"
    else:
        level = "Low"

    return {"score": score, "level": level}


def detect_emergency(selected_symptoms):
    """Checks selected symptoms against critical combination rules."""
    selected_set = set(selected_symptoms)
    rules = load_emergency_rules()
    for rule in rules:
        if rule.issubset(selected_set):
            return {
                "is_emergency": True,
                "triggered_by": sorted(rule),
            }
    return {"is_emergency": False, "triggered_by": []}


def get_recommendations(disease_name):
    """Looks up precautions, medication info, and recommended specialist."""
    precautions_map = _load_csv_dict("precautions.csv", "disease", "precautions", split_pipe=True)
    medications_map = _load_csv_dict("medications.csv", "disease", "medication_info")
    doctors_map = _load_csv_dict("doctors.csv", "disease", "specialist")

    default_medication = {
        "otc": [],
        "avoid": [],
        "prescription_note": "No specific information available — consult a doctor.",
        "urgent": False,
    }
    raw_medication = medications_map.get(disease_name)
    if raw_medication:
        try:
            medication_info = json.loads(raw_medication)
        except (ValueError, TypeError):
            medication_info = default_medication
    else:
        medication_info = default_medication

    return {
        "precautions": precautions_map.get(disease_name, ["Consult a healthcare professional."]),
        "medication_info": medication_info,
        "specialist": doctors_map.get(disease_name, "General Physician"),
    }


def run_pipeline(patient):
    """
    patient: dict with keys
        age (int), gender (str), weight_kg (float), height_cm (float),
        temperature_c (float), duration_days (int),
        existing_conditions (list[str]), symptoms (list[str])

    Returns a full result dict ready to render in the report / save to history.
    """
    symptoms = patient.get("symptoms", [])
    age = patient.get("age")
    temperature_c = patient.get("temperature_c")
    existing_conditions = patient.get("existing_conditions", [])

    bmi = None
    weight_kg = patient.get("weight_kg")
    height_cm = patient.get("height_cm")
    if weight_kg and height_cm:
        height_m = height_cm / 100
        if height_m > 0:
            bmi = round(weight_kg / (height_m ** 2), 1)

    emergency = detect_emergency(symptoms)
    predictions = predict_disease(symptoms, top_n=3)
    risk = calc_risk(age, temperature_c, existing_conditions, symptoms)

    # If even the top prediction has low confidence, the reported symptoms
    # likely don't closely match any of the ~40 conditions this model knows.
    # Rather than confidently naming a condition, flag this so the UI can
    # show an honest "this may be something else — see a doctor" notice.
    LOW_CONFIDENCE_THRESHOLD = 30.0  # percent
    low_confidence = bool(predictions) and predictions[0][1] < LOW_CONFIDENCE_THRESHOLD

    top_disease = predictions[0][0] if predictions else None
    recommendations = get_recommendations(top_disease) if top_disease else {
        "precautions": ["Please select at least one symptom to get a prediction."],
        "medication_info": {"otc": [], "avoid": [], "prescription_note": "N/A", "urgent": False},
        "specialist": "General Physician",
    }

    return {
        "bmi": bmi,
        "bmi_category": bmi_category(bmi),
        "emergency": emergency,
        "predictions": predictions,
        "risk": risk,
        "recommendations": recommendations,
        "top_disease": top_disease,
        "low_confidence": low_confidence,
    }

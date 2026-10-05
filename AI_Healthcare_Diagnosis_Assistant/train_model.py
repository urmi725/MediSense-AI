"""
train_model.py
----------------
Trains a RandomForestClassifier on the symptom -> disease dataset and saves:

  model/disease_model.pkl     - the trained classifier
  model/symptom_encoder.pkl   - ordered list of symptom columns (model input order)
  model/label_encoder.pkl     - sklearn LabelEncoder mapping disease <-> int
  model/metrics.json          - accuracy / CV score / per-class report / feature importances,
                                 read by the Analytics Dashboard in the web app

Prints held-out accuracy, a 5-fold stratified cross-validation score,
precision/recall/F1 per class, a confusion matrix, and the top symptoms
driving predictions (feature importance).

Run:  python train_model.py
"""

import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "diseases.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

N_ESTIMATORS = 180
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Run `python generate_dataset.py` first."
        )

    df = pd.read_csv(DATASET_PATH)
    symptom_columns = [c for c in df.columns if c != "Disease"]

    X = df[symptom_columns]
    y_raw = df["Disease"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=15,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("=" * 60)
    print(f"Held-out accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("=" * 60)

    # 5-fold stratified cross-validation on the full dataset for a more
    # robust accuracy estimate than a single train/test split.
    print(f"\nRunning {CV_FOLDS}-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X, y, cv=skf, n_jobs=-1)
    print(f"CV accuracy per fold: {[round(s, 4) for s in cv_scores]}")
    print(f"CV mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    print("\nClassification report (held-out test set):\n")
    report_text = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
    )
    print(report_text)
    report_dict = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0, output_dict=True
    )

    print("Confusion matrix (rows=true, cols=predicted):\n")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    print(cm_df.to_string())

    # Feature importance: which symptoms most influence the model's decisions
    importances = sorted(
        zip(symptom_columns, model.feature_importances_), key=lambda x: x[1], reverse=True
    )
    print("\nTop 15 most influential symptoms:\n")
    for name, score in importances[:15]:
        print(f"  {name:30s} {score:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "disease_model.pkl"))
    joblib.dump(symptom_columns, os.path.join(MODEL_DIR, "symptom_encoder.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    metrics = {
        "held_out_accuracy": round(acc, 4),
        "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "n_diseases": len(label_encoder.classes_),
        "n_symptoms": len(symptom_columns),
        "n_training_rows": len(df),
        "n_estimators": N_ESTIMATORS,
        "top_features": [{"symptom": n, "importance": round(float(s), 4)} for n, s in importances[:15]],
        "per_class_f1": {
            k: round(v["f1-score"], 4)
            for k, v in report_dict.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        },
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model artifacts + metrics.json to {MODEL_DIR}/")


if __name__ == "__main__":
    main()

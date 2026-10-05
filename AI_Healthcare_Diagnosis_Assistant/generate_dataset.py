"""
generate_dataset.py
--------------------
Generates a SYNTHETIC demo dataset for the AI Healthcare Diagnosis Assistant.

This does NOT use any real patient data. It samples symptoms for each disease
from rough illustrative probability profiles (dataset/disease_symptom_profiles.py),
adds noise, and writes out:

  dataset/symptoms.csv      - master symptom list (name, display name, severity)
  dataset/diseases.csv      - multi-hot symptom vectors -> Disease label (training data)
  dataset/precautions.csv   - Disease -> precaution advice
  dataset/medications.csv   - Disease -> informational medication notes
  dataset/doctors.csv       - Disease -> recommended specialist
  dataset/severity.csv      - symptom -> emergency flag (used in emergency rules)

Run:  python generate_dataset.py
"""

import csv
import random
import os
import json

from dataset.disease_symptom_profiles import (
    DISEASE_SYMPTOM_PROFILES,
    SYMPTOM_META,
    PRECAUTIONS,
    MEDICATIONS,
    DOCTORS,
    EMERGENCY_RULES,
)

RANDOM_SEED = 42
ROWS_PER_DISEASE = 260          # 40 diseases * 260 = ~10,400 rows
NOISE_FLIP_PROB = 0.03          # chance a symptom bit gets randomly flipped
EXTRA_RANDOM_SYMPTOM_PROB = 0.02  # chance of an unrelated symptom appearing

OUT_DIR = os.path.join(os.path.dirname(__file__), "dataset")


def build_symptom_list():
    """Union of every symptom referenced anywhere, alphabetically sorted."""
    symptoms = set(SYMPTOM_META.keys())
    for profile in DISEASE_SYMPTOM_PROFILES.values():
        symptoms.update(profile.keys())
    return sorted(symptoms)


def write_symptoms_csv(symptom_list):
    path = os.path.join(OUT_DIR, "symptoms.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["symptom", "display_name", "severity_weight"])
        for s in symptom_list:
            display, weight = SYMPTOM_META.get(s, (s.replace("_", " ").title(), 1))
            writer.writerow([s, display, weight])
    print(f"Wrote {path} ({len(symptom_list)} symptoms)")


def write_severity_csv(symptom_list):
    """Flags which individual symptoms count as 'high severity' on their own."""
    path = os.path.join(OUT_DIR, "severity.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["symptom", "severity_weight", "is_emergency_flag"])
        emergency_symptoms = set()
        for combo in EMERGENCY_RULES:
            emergency_symptoms.update(combo)
        for s in symptom_list:
            _, weight = SYMPTOM_META.get(s, (s, 1))
            writer.writerow([s, weight, int(s in emergency_symptoms)])
    print(f"Wrote {path}")


def write_lookup_csv(filename, mapping, value_header, is_list=False, is_json=False):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["disease", value_header])
        for disease, value in mapping.items():
            if is_list:
                value = " | ".join(value)
            elif is_json:
                value = json.dumps(value)
            writer.writerow([disease, value])
    print(f"Wrote {path}")


def generate_diseases_csv(symptom_list):
    random.seed(RANDOM_SEED)
    path = os.path.join(OUT_DIR, "diseases.csv")
    diseases = list(DISEASE_SYMPTOM_PROFILES.keys())

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(symptom_list + ["Disease"])

        total_rows = 0
        for disease in diseases:
            profile = DISEASE_SYMPTOM_PROFILES[disease]
            for _ in range(ROWS_PER_DISEASE):
                row = []
                for s in symptom_list:
                    base_prob = profile.get(s, EXTRA_RANDOM_SYMPTOM_PROB)
                    present = random.random() < base_prob
                    # noise: small chance to flip the bit either way
                    if random.random() < NOISE_FLIP_PROB:
                        present = not present
                    row.append(int(present))
                row.append(disease)
                writer.writerow(row)
                total_rows += 1

    print(f"Wrote {path} ({total_rows} rows, {len(symptom_list)} symptom columns, {len(diseases)} diseases)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    symptom_list = build_symptom_list()

    write_symptoms_csv(symptom_list)
    write_severity_csv(symptom_list)
    write_lookup_csv("precautions.csv", PRECAUTIONS, "precautions", is_list=True)
    write_lookup_csv("medications.csv", MEDICATIONS, "medication_info", is_json=True)
    write_lookup_csv("doctors.csv", DOCTORS, "specialist")
    generate_diseases_csv(symptom_list)

    print("\nDataset generation complete.")
    print("NOTE: This is a SYNTHETIC demo dataset only — not real clinical data.")


if __name__ == "__main__":
    main()

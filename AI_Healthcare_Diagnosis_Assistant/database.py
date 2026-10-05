"""
database.py
------------
Local SQLite storage for user accounts, patient checks / prediction history,
and per-report feedback. No external transmission — the database file
(healthcare.db) is created automatically in the project root on first run.

Tables:
  users        - simple username/password (hashed) accounts
  predictions  - every symptom check + its result, optionally tied to a user
  feedback     - "was this helpful?" responses tied to a prediction
"""

import os
import sqlite3
import json
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "healthcare.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Creates all tables if they don't already exist."""
    conn = get_connection()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            created_at TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            weight_kg REAL,
            height_cm REAL,
            bmi REAL,
            bmi_category TEXT,
            temperature_c REAL,
            duration_days INTEGER,
            existing_conditions TEXT,
            symptoms TEXT,
            top_disease TEXT,
            confidence REAL,
            risk_level TEXT,
            is_emergency INTEGER,
            specialist TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            is_helpful INTEGER NOT NULL,
            comment TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (prediction_id) REFERENCES predictions (id)
        )
        """
    )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------- users ----

def create_user(username, password, full_name=None):
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, full_name, created_at) VALUES (?, ?, ?, ?)",
            (username.strip(), generate_password_hash(password), full_name, datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
        user_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
        return user_id
    except sqlite3.IntegrityError:
        return None  # username already taken
    finally:
        conn.close()


def get_user_by_username(username):
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username.strip(),)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def verify_password(user, password):
    return check_password_hash(user["password_hash"], password)


# ----------------------------------------------------------- predictions ----

def save_prediction(patient, result, user_id=None):
    """Saves a single patient check + result to the database."""
    conn = get_connection()
    top_pred = result["predictions"][0] if result["predictions"] else (None, None)
    conn.execute(
        """
        INSERT INTO predictions (
            user_id, created_at, age, gender, weight_kg, height_cm, bmi, bmi_category,
            temperature_c, duration_days, existing_conditions, symptoms,
            top_disease, confidence, risk_level, is_emergency, specialist
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            datetime.now().isoformat(timespec="seconds"),
            patient.get("age"),
            patient.get("gender"),
            patient.get("weight_kg"),
            patient.get("height_cm"),
            result.get("bmi"),
            result.get("bmi_category"),
            patient.get("temperature_c"),
            patient.get("duration_days"),
            json.dumps(patient.get("existing_conditions", [])),
            json.dumps(patient.get("symptoms", [])),
            top_pred[0],
            top_pred[1],
            result["risk"]["level"],
            int(result["emergency"]["is_emergency"]),
            result["recommendations"]["specialist"],
        ),
    )
    conn.commit()
    row_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    conn.close()
    return row_id


def _row_to_record(row):
    record = dict(row)
    record["existing_conditions"] = json.loads(record["existing_conditions"] or "[]")
    record["symptoms"] = json.loads(record["symptoms"] or "[]")
    return record


def get_history(user_id=None, limit=200):
    """
    Returns the most recent predictions, newest first.
    Logged-in users only see their own records; guests only see other
    guest (unowned) records, never another logged-in user's history.
    """
    conn = get_connection()
    if user_id is not None:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE user_id IS NULL ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [_row_to_record(r) for r in rows]


def get_prediction_by_id(prediction_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
    conn.close()
    return _row_to_record(row) if row else None


def clear_history(user_id=None):
    """Clears only the caller's own records — a logged-in user's, or unowned guest records."""
    conn = get_connection()
    if user_id is not None:
        conn.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
    else:
        conn.execute("DELETE FROM predictions WHERE user_id IS NULL")
    conn.commit()
    conn.close()


# -------------------------------------------------------------- feedback ----

def save_feedback(prediction_id, is_helpful, comment=None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO feedback (prediction_id, is_helpful, comment, created_at) VALUES (?, ?, ?, ?)",
        (prediction_id, int(is_helpful), comment, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def get_feedback_stats():
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) AS total, SUM(is_helpful) AS helpful FROM feedback"
    ).fetchone()
    conn.close()
    total = row["total"] or 0
    helpful = row["helpful"] or 0
    pct = round((helpful / total) * 100, 1) if total else None
    return {"total": total, "helpful": helpful, "helpful_pct": pct}


# -------------------------------------------------------------- analytics ----

def get_dashboard_stats(user_id=None):
    """
    Aggregate stats for the Analytics Dashboard:
      - disease frequency (top predicted conditions)
      - risk level distribution
      - most commonly reported symptoms
      - totals (checks, emergencies flagged)
    If user_id is given, stats are scoped to that user only.
    """
    conn = get_connection()
    where = "WHERE user_id = ?" if user_id is not None else ""
    params = (user_id,) if user_id is not None else ()

    total_checks = conn.execute(
        f"SELECT COUNT(*) AS c FROM predictions {where}", params
    ).fetchone()["c"]

    emergency_count = conn.execute(
        f"SELECT COUNT(*) AS c FROM predictions {where} {'AND' if where else 'WHERE'} is_emergency = 1",
        params,
    ).fetchone()["c"]

    disease_rows = conn.execute(
        f"""
        SELECT top_disease, COUNT(*) AS c FROM predictions
        {where} {'AND' if where else 'WHERE'} top_disease IS NOT NULL
        GROUP BY top_disease ORDER BY c DESC LIMIT 10
        """,
        params,
    ).fetchall()

    risk_rows = conn.execute(
        f"""
        SELECT risk_level, COUNT(*) AS c FROM predictions
        {where} {'AND' if where else 'WHERE'} risk_level IS NOT NULL
        GROUP BY risk_level
        """,
        params,
    ).fetchall()

    symptom_rows = conn.execute(f"SELECT symptoms FROM predictions {where}", params).fetchall()

    conn.close()

    symptom_counter = {}
    for row in symptom_rows:
        for s in json.loads(row["symptoms"] or "[]"):
            symptom_counter[s] = symptom_counter.get(s, 0) + 1
    top_symptoms = sorted(symptom_counter.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total_checks": total_checks,
        "emergency_count": emergency_count,
        "disease_counts": [{"label": r["top_disease"], "count": r["c"]} for r in disease_rows],
        "risk_counts": {r["risk_level"]: r["c"] for r in risk_rows},
        "top_symptoms": [{"label": s.replace("_", " "), "count": c} for s, c in top_symptoms],
    }

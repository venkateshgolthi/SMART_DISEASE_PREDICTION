"""SQLite storage for user accounts and prediction history."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "app.db"


@contextmanager
def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT NOT NULL,
                email         TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id           INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at        TEXT NOT NULL,
                patient_ref       TEXT NOT NULL,
                age               INTEGER,
                weight_kg         REAL,
                height_cm         REAL,
                bmi               REAL,
                symptoms          TEXT NOT NULL,
                predicted_disease TEXT NOT NULL,
                model_used        TEXT NOT NULL,
                model_predictions TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_pred_user ON predictions(user_id, created_at);
            """
        )


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ------------------------- users -------------------------
def create_user(name: str, email: str, password_hash: str) -> bool:
    """Returns False if the e-mail is already registered."""
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO users (name, email, password_hash, created_at) VALUES (?,?,?,?)",
                (name, email, password_hash, _now()),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def get_user_by_email(email: str):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    return dict(row) if row else None


# ----------------------- predictions ----------------------
def add_prediction(user_id: int, rec: dict) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO predictions
               (user_id, created_at, patient_ref, age, weight_kg, height_cm, bmi,
                symptoms, predicted_disease, model_used, model_predictions)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                user_id, _now(), rec["patient_ref"], rec.get("age"), rec.get("weight_kg"),
                rec.get("height_cm"), rec.get("bmi"), json.dumps(rec["symptoms"]),
                rec["predicted_disease"], rec["model_used"],
                json.dumps(rec.get("individual") or {}),
            ),
        )
        return cur.lastrowid


def get_predictions(user_id: int) -> list:
    """Only ever returns rows that belong to `user_id`."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC, id DESC",
            (user_id,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["symptoms"] = json.loads(d["symptoms"])
        d["model_predictions"] = json.loads(d["model_predictions"] or "{}")
        out.append(d)
    return out

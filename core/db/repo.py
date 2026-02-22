"""CRUD operations for all tables: users, runs, batches, batch_runs, audit_events."""
import json
import sqlite3
import bcrypt
from typing import Optional

from core.util.ids import new_user_id, new_run_id, new_batch_id, new_event_id
from core.util.time import utcnow_iso


# ─────────────────────────────── USERS ────────────────────────────────────

def create_user(conn: sqlite3.Connection, email: str, display_name: str, password: str) -> str:
    user_id = new_user_id()
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    conn.execute(
        "INSERT INTO users (user_id, email, display_name, password_hash, created_at) VALUES (?,?,?,?,?)",
        (user_id, email, display_name, pw_hash, utcnow_iso()),
    )
    conn.commit()
    return user_id


def authenticate_user(conn: sqlite3.Connection, email: str, password: str) -> Optional[str]:
    row = conn.execute("SELECT user_id, password_hash FROM users WHERE email=?", (email,)).fetchone()
    if row is None:
        return None
    if bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
        return row["user_id"]
    return None


def get_user_display_name(conn: sqlite3.Connection, user_id: str) -> Optional[str]:
    row = conn.execute("SELECT display_name FROM users WHERE user_id=?", (user_id,)).fetchone()
    return row["display_name"] if row else None


# ─────────────────────────────── RUNS ─────────────────────────────────────

def create_run(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    image_hash: str,
    report_hash: str,
    case_label: str = "",
    model_name: str,
    model_version: str,
    lora_id: str = "",
    prompt_version: str,
    overall_score: int,
    severity: str,
    flag_counts: dict,
    status: str = "complete",
    error_message: str = "",
    results_path: str,
) -> str:
    run_id = new_run_id()
    conn.execute(
        """INSERT INTO runs
        (run_id, user_id, created_at, input_image_hash, input_report_hash,
         case_label, model_name, model_version, lora_id, prompt_version,
         overall_score, severity, flag_counts_json, status, error_message, results_path)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            run_id, user_id, utcnow_iso(), image_hash, report_hash,
            case_label, model_name, model_version, lora_id or "", prompt_version,
            overall_score, severity, json.dumps(flag_counts),
            status, error_message or "", results_path,
        ),
    )
    conn.commit()
    return run_id


def get_run(conn: sqlite3.Connection, run_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["flag_counts"] = json.loads(d.pop("flag_counts_json"))
    return d


def list_recent_runs_for_user(conn: sqlite3.Connection, user_id: str, limit: int = 20) -> list:
    rows = conn.execute(
        "SELECT created_at, case_label, overall_score, severity, run_id "
        "FROM runs WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return [list(r) for r in rows]


def list_all_runs_for_user(conn: sqlite3.Connection, user_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM runs WHERE user_id=? ORDER BY created_at DESC", (user_id,)
    ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["flag_counts"] = json.loads(d.pop("flag_counts_json"))
        result.append(d)
    return result


# ─────────────────────────────── BATCHES ──────────────────────────────────

def create_batch(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    zip_name: str,
    num_cases_total: int,
) -> str:
    batch_id = new_batch_id()
    conn.execute(
        """INSERT INTO batches
        (batch_id, user_id, created_at, zip_name, num_cases_total,
         num_cases_done, num_cases_failed, batch_summary_json, status)
        VALUES (?,?,?,?,?,0,0,'{}','running')""",
        (batch_id, user_id, utcnow_iso(), zip_name, num_cases_total),
    )
    conn.commit()
    return batch_id


def update_batch_progress(
    conn: sqlite3.Connection,
    batch_id: str,
    num_done: int,
    num_failed: int,
    summary: dict,
    status: str = "running",
) -> None:
    conn.execute(
        """UPDATE batches SET num_cases_done=?, num_cases_failed=?,
           batch_summary_json=?, status=? WHERE batch_id=?""",
        (num_done, num_failed, json.dumps(summary), status, batch_id),
    )
    conn.commit()


def link_batch_run(conn: sqlite3.Connection, batch_id: str, run_id: str, case_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO batch_runs (batch_id, run_id, case_id) VALUES (?,?,?)",
        (batch_id, run_id, case_id),
    )
    conn.commit()


def get_batch(conn: sqlite3.Connection, batch_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["batch_summary"] = json.loads(d.pop("batch_summary_json"))
    return d


def list_batch_runs(conn: sqlite3.Connection, batch_id: str) -> list[dict]:
    rows = conn.execute(
        """SELECT r.*, br.case_id FROM runs r
           JOIN batch_runs br ON r.run_id=br.run_id
           WHERE br.batch_id=? ORDER BY r.created_at""",
        (batch_id,),
    ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["flag_counts"] = json.loads(d.pop("flag_counts_json"))
        result.append(d)
    return result


# ─────────────────────────── AUDIT EVENTS ────────────────────────────────

def log_event(
    conn: sqlite3.Connection,
    run_id: str,
    actor: str,
    event_type: str,
    details: dict,
) -> str:
    event_id = new_event_id()
    conn.execute(
        "INSERT INTO audit_events (event_id, run_id, timestamp, actor, event_type, details_json) VALUES (?,?,?,?,?,?)",
        (event_id, run_id, utcnow_iso(), actor, event_type, json.dumps(details)),
    )
    conn.commit()
    return event_id


def list_events_for_run(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM audit_events WHERE run_id=? ORDER BY timestamp", (run_id,)
    ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["details"] = json.loads(d.pop("details_json"))
        result.append(d)
    return result

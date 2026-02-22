"""Thread-safe SQLite connection management."""
import sqlite3
import threading
from pathlib import Path

_local = threading.local()


def get_conn(db_path: Path) -> sqlite3.Connection:
    """Return a thread-local connection to the SQLite database."""
    key = str(db_path)
    conn = getattr(_local, key, None)
    if conn is None:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        setattr(_local, key, conn)
    return conn


def init_db(db_path: Path, schema_path: Path) -> None:
    """Create tables from schema SQL if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_conn(db_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()

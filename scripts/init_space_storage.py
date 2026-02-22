from pathlib import Path
import sqlite3

def ensure_space_storage(storage_dir: Path, db_path: Path) -> None:
    """
    Hugging Face Spaces: create a writable storage area for sqlite + outputs.
    """
    storage_dir.mkdir(parents=True, exist_ok=True)

    # outputs dirs (inside storage so it persists for the app runtime)
    (storage_dir / "outputs" / "runs").mkdir(parents=True, exist_ok=True)
    (storage_dir / "outputs" / "batches").mkdir(parents=True, exist_ok=True)

    # db init
    if not db_path.exists():
        schema_path = Path("core/db/schema.sql")
        if not schema_path.exists():
            raise FileNotFoundError("core/db/schema.sql not found")

        conn = sqlite3.connect(db_path)
        with open(schema_path, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()
        conn.close()

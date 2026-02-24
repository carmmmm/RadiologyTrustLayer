"""File I/O helpers for run outputs."""
import json
import shutil
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist, then return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: dict) -> None:
    """Write a dict as pretty-printed JSON, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> dict:
    """Read and parse a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: Path) -> str:
    """Read a file as UTF-8 text."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def copy_file(src: Path, dst: Path) -> None:
    """Copy a file preserving metadata, creating destination directories as needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

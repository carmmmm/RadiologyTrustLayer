"""
Parse a ZIP archive of radiology cases into a list of (case_id, image_path, report_text) tuples.

Expected ZIP structure (either flat or one-folder-per-case):

Flat (image+report share same stem):
  archive.zip/
    case01.png
    case01.txt
    case02.jpg
    case02.txt

Per-folder:
  archive.zip/
    case01/
      image.png   (or image.jpg / scan.png / etc.)
      report.txt
    case02/
      image.jpg
      report.txt
"""
import zipfile
from pathlib import Path
from typing import NamedTuple

from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
REPORT_NAMES = {"report.txt", "report.md", "findings.txt", "text.txt"}


class CaseInput(NamedTuple):
    case_id: str
    image: Image.Image
    report_text: str


def parse_zip(zip_path: Path, extract_dir: Path) -> list[CaseInput]:
    """
    Extract and parse a ZIP archive into CaseInput list.
    Raises ValueError if no valid cases are found.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    cases: list[CaseInput] = []

    # Strategy 1: per-folder layout
    for folder in sorted(extract_dir.iterdir()):
        if not folder.is_dir():
            continue
        img_path = _find_image(folder)
        rpt_path = _find_report(folder)
        if img_path and rpt_path:
            try:
                img = Image.open(img_path).convert("RGB")
                text = rpt_path.read_text(encoding="utf-8", errors="replace")
                cases.append(CaseInput(case_id=folder.name, image=img, report_text=text.strip()))
            except Exception:
                pass

    # Strategy 2: flat layout (image+report share same stem)
    if not cases:
        files_by_stem: dict[str, dict] = {}
        for f in extract_dir.iterdir():
            if f.is_file():
                stem = f.stem
                ext = f.suffix.lower()
                if ext in IMAGE_EXTS:
                    files_by_stem.setdefault(stem, {})["image"] = f
                elif ext in {".txt", ".md"}:
                    files_by_stem.setdefault(stem, {})["report"] = f

        for stem, parts in sorted(files_by_stem.items()):
            if "image" in parts and "report" in parts:
                try:
                    img = Image.open(parts["image"]).convert("RGB")
                    text = parts["report"].read_text(encoding="utf-8", errors="replace")
                    cases.append(CaseInput(case_id=stem, image=img, report_text=text.strip()))
                except Exception:
                    pass

    if not cases:
        raise ValueError(
            "No valid cases found in ZIP. Each case needs an image file "
            "(.png/.jpg/etc.) paired with a report (.txt)."
        )

    return cases


def _find_image(folder: Path) -> Path | None:
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            return f
    return None


def _find_report(folder: Path) -> Path | None:
    for name in REPORT_NAMES:
        p = folder / name
        if p.exists():
            return p
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in {".txt", ".md"}:
            return f
    return None

"""Sequential batch runner: runs audit_pipeline on each case and persists batch results."""
import logging
from pathlib import Path
from typing import Callable, Optional

from core.batch.parse_zip import parse_zip, CaseInput
from core.pipeline.audit_pipeline import run_audit
from core.util.files import write_json
from core import config

logger = logging.getLogger(__name__)


def run_batch(
    zip_path: Path,
    extract_dir: Path,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """
    Run the full audit pipeline on each case in the ZIP.

    Returns a batch_result dict with per-case results and summary statistics.
    """
    cases: list[CaseInput] = parse_zip(zip_path, extract_dir)
    total = len(cases)
    results = []
    errors = []

    for i, case in enumerate(cases, start=1):
        if progress_cb:
            progress_cb(i, total, f"Auditing case {i}/{total}: {case.case_id}")

        try:
            def case_progress(step, total_steps, msg):
                if progress_cb:
                    detail = f"[{case.case_id}] {msg}"
                    progress_cb(i, total, detail)

            result = run_audit(
                image=case.image,
                report_text=case.report_text,
                case_label=case.case_id,
                progress_cb=case_progress,
            )
            results.append(result)
        except Exception as e:
            logger.error("Case %s failed: %s", case.case_id, e)
            errors.append({"case_id": case.case_id, "error": str(e)})

    # Compute batch summary
    scores = [r["overall_score"] for r in results]
    severities = [r["severity"] for r in results]

    summary = {
        "total_cases": total,
        "completed": len(results),
        "failed": len(errors),
        "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
        "severity_distribution": {
            "low": severities.count("low"),
            "medium": severities.count("medium"),
            "high": severities.count("high"),
        },
        "pct_needing_review": round(
            sum(1 for r in results if r["severity"] in ("medium", "high")) / max(len(results), 1) * 100, 1
        ),
        "errors": errors,
    }

    return {
        "total": total,
        "results": results,
        "summary": summary,
    }

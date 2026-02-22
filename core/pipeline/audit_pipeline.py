"""
Main audit orchestrator.

Runs the full 6-step pipeline:
  1. Claim extraction (text)
  2. Image findings (image)
  3. Alignment (text + prior results)
  4. Rewrite suggestions (text + alignment)
  5. Clinician summary (text + scoring)
  6. Patient explanation (text)

Returns a fully structured AuditResult dict and persists it to disk.
"""
import json
import logging
from pathlib import Path
from typing import Optional, Generator

from PIL import Image

from core import config
from core.pipeline import medgemma_client as mgc
from core.scoring.score import compute_score
from core.util.files import write_json, read_text
from core.util.hashing import hash_image, hash_string
from core.util.ids import new_run_id
from core.util.time import utcnow_iso

logger = logging.getLogger(__name__)

SCHEMAS = config.SCHEMAS_DIR
PROMPTS = config.PROMPTS_DIR


def _load_prompt(name: str, **kwargs) -> str:
    path = PROMPTS / f"{name}.md"
    template = read_text(path)
    return template.format(**kwargs)


def run_audit(
    image: Image.Image,
    report_text: str,
    case_label: str = "",
    lora_id: str = "",
    progress_cb: Optional[callable] = None,
) -> dict:
    """
    Run the full audit pipeline.

    Args:
        image: PIL Image (radiology scan)
        report_text: Free-text radiology report
        case_label: Optional label for this case
        lora_id: Optional HF repo id of LoRA adapter to use
        progress_cb: Optional callable(step: int, total: int, message: str)

    Returns:
        AuditResult dict (also saved to disk)
    """
    run_id = new_run_id()
    total_steps = 6
    errors: list[str] = []
    schema_repairs: list[str] = []

    # Set mock context so mock mode returns case-specific results
    mgc.set_mock_context(report_text)

    def progress(step: int, msg: str):
        logger.info("[%s] Step %d/%d: %s", run_id, step, total_steps, msg)
        if progress_cb:
            progress_cb(step, total_steps, msg)

    # ── Step 1: Claim extraction ─────────────────────────────────────────
    progress(1, "Extracting claims from report...")
    prompt = _load_prompt(
        "claim_extraction",
        report_text=report_text,
        schema=open(SCHEMAS / "claim_extraction.schema.json").read(),
    )
    claims_result, errs = mgc.infer_structured(
        prompt=prompt,
        schema_path=SCHEMAS / "claim_extraction.schema.json",
        image=None,
        task_name="claim_extraction",
    )
    if errs:
        errors.extend(errs)
        schema_repairs.append("claim_extraction")
    claims = claims_result.get("claims", [])

    # ── Step 2: Image findings ────────────────────────────────────────────
    progress(2, "Analyzing image for visual findings...")
    prompt = _load_prompt(
        "image_findings",
        schema=open(SCHEMAS / "image_findings.schema.json").read(),
    )
    findings_result, errs = mgc.infer_structured(
        prompt=prompt,
        schema_path=SCHEMAS / "image_findings.schema.json",
        image=image,
        task_name="image_findings",
    )
    if errs:
        errors.extend(errs)
        schema_repairs.append("image_findings")
    findings = findings_result.get("findings", [])
    image_quality = findings_result.get("image_quality", "adequate")

    # ── Step 3: Alignment ─────────────────────────────────────────────────
    progress(3, "Aligning report claims to image evidence...")
    prompt = _load_prompt(
        "alignment",
        claims_json=json.dumps(claims, indent=2),
        findings_json=json.dumps(findings, indent=2),
        schema=open(SCHEMAS / "alignment.schema.json").read(),
    )
    alignment_result, errs = mgc.infer_structured(
        prompt=prompt,
        schema_path=SCHEMAS / "alignment.schema.json",
        image=None,
        task_name="alignment",
    )
    if errs:
        errors.extend(errs)
        schema_repairs.append("alignment")
    alignments = alignment_result.get("alignments", [])

    # Merge claim text into alignments for convenience
    claim_map = {c["claim_id"]: c for c in claims}
    for a in alignments:
        cid = a.get("claim_id", "")
        a["claim_text"] = claim_map.get(cid, {}).get("text", "")

    # ── Step 4: Scoring ───────────────────────────────────────────────────
    progress(4, "Computing safety score...")
    overall_score, severity, flag_counts = compute_score(alignments)

    # ── Step 5: Rewrite suggestions ───────────────────────────────────────
    progress(5, "Generating rewrite suggestions...")
    flagged = [a for a in alignments if a.get("label") in ("uncertain", "needs_review")]
    prompt = _load_prompt(
        "rewrite",
        report_text=report_text,
        alignment_json=json.dumps(flagged, indent=2),
        schema=open(SCHEMAS / "rewrite.schema.json").read(),
    )
    rewrite_result, errs = mgc.infer_structured(
        prompt=prompt,
        schema_path=SCHEMAS / "rewrite.schema.json",
        image=None,
        task_name="rewrite",
    )
    if errs:
        errors.extend(errs)
        schema_repairs.append("rewrite")

    # ── Step 5b: Clinician summary ────────────────────────────────────────
    flagged_claims = [a for a in alignments if a.get("label") in ("uncertain", "needs_review", "not_assessable")]
    prompt = _load_prompt(
        "clinician_summary",
        overall_score=overall_score,
        severity=severity,
        flag_counts_json=json.dumps(flag_counts),
        flagged_claims_json=json.dumps(flagged_claims, indent=2),
        schema=open(SCHEMAS / "clinician_summary.schema.json").read(),
    )
    summary_result, errs = mgc.infer_structured(
        prompt=prompt,
        schema_path=SCHEMAS / "clinician_summary.schema.json",
        image=None,
        task_name="clinician_summary",
    )
    if errs:
        errors.extend(errs)
        schema_repairs.append("clinician_summary")

    # ── Step 6: Patient explanation ───────────────────────────────────────
    progress(6, "Generating patient-friendly explanation...")
    edited_report = rewrite_result.get("edited_report", report_text)
    prompt = _load_prompt(
        "patient_explain",
        report_text=edited_report,
        schema=open(SCHEMAS / "patient_explain.schema.json").read(),
    )
    patient_result, errs = mgc.infer_structured(
        prompt=prompt,
        schema_path=SCHEMAS / "patient_explain.schema.json",
        image=None,
        task_name="patient_explain",
    )
    if errs:
        errors.extend(errs)
        schema_repairs.append("patient_explain")

    # ── Assemble result ───────────────────────────────────────────────────
    result = {
        "run_id": run_id,
        "created_at": utcnow_iso(),
        "case_label": case_label,
        "model_name": "medgemma",
        "model_version": config.MEDGEMMA_MODEL_ID,
        "lora_id": lora_id or config.RTL_LORA_ID,
        "prompt_version": config.RTL_PROMPT_VERSION,
        "mock_mode": config.MEDGEMMA_MOCK,
        "image_hash": hash_image(image),
        "report_hash": hash_string(report_text),
        "original_report": report_text,
        "claims": claims,
        "findings": findings,
        "image_quality": image_quality,
        "alignments": alignments,
        "overall_score": overall_score,
        "severity": severity,
        "flag_counts": flag_counts,
        "rewrites": rewrite_result.get("rewrites", []),
        "edited_report": edited_report,
        "clinician_summary": summary_result,
        "patient_explanation": patient_result,
        "pipeline_errors": errors,
        "schema_repairs": schema_repairs,
    }

    # ── Persist to disk ───────────────────────────────────────────────────
    run_dir = config.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.json"
    write_json(results_path, result)
    result["results_path"] = str(results_path)

    logger.info("[%s] Audit complete. Score=%d Severity=%s", run_id, overall_score, severity)
    return result

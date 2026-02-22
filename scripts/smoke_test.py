"""
Smoke test: run the full pipeline on an example case with mock mode.
Validates that all components are wired up correctly.

Usage:
  MEDGEMMA_MOCK=true python scripts/smoke_test.py
"""
import json
import sys
import os
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MEDGEMMA_MOCK", "true")


def test_pipeline():
    from PIL import Image
    import numpy as np

    # Create a dummy grayscale "X-ray" image
    img_array = np.random.randint(50, 200, (512, 512), dtype=np.uint8)
    image = Image.fromarray(img_array, mode="L").convert("RGB")

    report = (
        "There is consolidation in the right lower lobe consistent with pneumonia. "
        "No pleural effusion is identified. "
        "The cardiomediastinal silhouette is within normal limits. "
        "No pneumothorax is seen. "
        "Mild hyperinflation is noted, possibly consistent with early COPD."
    )

    from core.pipeline.audit_pipeline import run_audit

    steps_logged = []
    def progress_cb(step, total, msg):
        steps_logged.append((step, total, msg))
        print(f"  [{step}/{total}] {msg}")

    print("Running audit pipeline (mock mode)...")
    result = run_audit(image=image, report_text=report, case_label="smoke_test", progress_cb=progress_cb)

    # Assertions
    assert result["overall_score"] >= 0, "Score must be >= 0"
    assert result["overall_score"] <= 100, "Score must be <= 100"
    assert result["severity"] in ("low", "medium", "high"), f"Bad severity: {result['severity']}"
    assert len(result["claims"]) > 0, "No claims extracted"
    assert len(result["alignments"]) > 0, "No alignments"
    assert "rewrites" in result, "Missing rewrites"
    assert "clinician_summary" in result, "Missing clinician_summary"
    assert "patient_explanation" in result, "Missing patient_explanation"
    assert "edited_report" in result, "Missing edited_report"
    assert len(steps_logged) == 6, f"Expected 6 progress steps, got {len(steps_logged)}"

    print(f"\n✅ Pipeline OK — Score: {result['overall_score']}/100, Severity: {result['severity']}")
    return result


def test_database():
    from core import config
    from core.db.db import get_conn, init_db
    from core.db.repo import create_user, authenticate_user, create_run, list_recent_runs_for_user

    # Use temp DB
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        schema_path = ROOT / "core" / "db" / "schema.sql"
        init_db(db_path, schema_path)
        conn = get_conn(db_path)

        # Test user creation
        uid = create_user(conn, "test@test.com", "Test User", "password123")
        assert uid, "No user_id returned"

        # Test authentication
        authed = authenticate_user(conn, "test@test.com", "password123")
        assert authed == uid, "Auth failed"

        # Test wrong password
        bad_auth = authenticate_user(conn, "test@test.com", "wrongpassword")
        assert bad_auth is None, "Bad auth should return None"

        # Test run creation
        run_id = create_run(
            conn, user_id=uid,
            image_hash="abc123", report_hash="def456",
            case_label="test_case", model_name="medgemma",
            model_version="google/medgemma-4b-it", prompt_version="v1",
            overall_score=85, severity="low", flag_counts={"supported":4,"uncertain":1},
            results_path="/tmp/results.json",
        )
        assert run_id, "No run_id returned"

        # Test listing
        rows = list_recent_runs_for_user(conn, uid)
        assert len(rows) == 1, f"Expected 1 run, got {len(rows)}"

    print("✅ Database OK")


def test_scoring():
    from core.scoring.score import compute_score

    claims = [
        {"label": "supported"},
        {"label": "supported"},
        {"label": "uncertain"},
        {"label": "needs_review"},
    ]
    score, severity, counts = compute_score(claims)
    assert 0 <= score <= 100
    assert severity in ("low", "medium", "high")
    assert counts["supported"] == 2
    assert counts["uncertain"] == 1
    assert counts["needs_review"] == 1

    # Perfect score
    perfect_claims = [{"label": "supported"} for _ in range(5)]
    score2, sev2, _ = compute_score(perfect_claims)
    assert score2 == 100, f"Expected 100, got {score2}"
    assert sev2 == "low"

    print("✅ Scoring OK")


def test_validation():
    from core.util.validation import extract_json_from_text, validate, load_schema

    # Test extraction
    text = 'Some text {"key": "value", "num": 42} more text'
    result = extract_json_from_text(text)
    assert result == {"key": "value", "num": 42}, f"Wrong result: {result}"

    # Test JSON block extraction
    text2 = '```json\n{"claims": []}\n```'
    result2 = extract_json_from_text(text2)
    assert result2 == {"claims": []}, f"Wrong result: {result2}"

    print("✅ Validation OK")


if __name__ == "__main__":
    print("=" * 50)
    print("RTL Smoke Test")
    print("=" * 50)

    try:
        test_scoring()
        test_validation()
        test_database()
        test_pipeline()
        print("\n🎉 All smoke tests passed!")
    except Exception as e:
        print(f"\n❌ Smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

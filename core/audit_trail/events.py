"""Audit event type constants and helper to log pipeline steps."""
from enum import Enum
from sqlite3 import Connection

from core.db.repo import log_event


class EventType(str, Enum):
    # Pipeline steps
    PIPELINE_START = "pipeline.start"
    CLAIM_EXTRACTION = "pipeline.claim_extraction"
    IMAGE_FINDINGS = "pipeline.image_findings"
    ALIGNMENT = "pipeline.alignment"
    REWRITE = "pipeline.rewrite"
    CLINICIAN_SUMMARY = "pipeline.clinician_summary"
    PATIENT_EXPLAIN = "pipeline.patient_explain"
    SCORING = "pipeline.scoring"
    PIPELINE_COMPLETE = "pipeline.complete"
    PIPELINE_ERROR = "pipeline.error"
    SCHEMA_REPAIR = "pipeline.schema_repair"

    # User actions
    USER_ACCEPT_REWRITE = "user.accept_rewrite"
    USER_REJECT_REWRITE = "user.reject_rewrite"
    USER_EXPORT = "user.export"
    USER_VIEW = "user.view"

    # Batch
    BATCH_START = "batch.start"
    BATCH_CASE_DONE = "batch.case_done"
    BATCH_CASE_FAILED = "batch.case_failed"
    BATCH_COMPLETE = "batch.complete"


def log(conn: Connection, run_id: str, event_type: EventType, details: dict, actor: str = "system") -> str:
    return log_event(conn, run_id, actor, event_type.value, details)

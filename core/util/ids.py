"""Unique ID generation for runs, batches, events, and users."""
import uuid
import time


def new_id(prefix: str = "") -> str:
    """Return a collision-resistant ID with an optional prefix."""
    ts = int(time.time() * 1000)
    uid = uuid.uuid4().hex[:12]
    if prefix:
        return f"{prefix}_{ts}_{uid}"
    return f"{ts}_{uid}"


def new_run_id() -> str:
    """Generate a unique run identifier (e.g. run_1700000000000_abc123)."""
    return new_id("run")


def new_batch_id() -> str:
    """Generate a unique batch identifier."""
    return new_id("bat")


def new_event_id() -> str:
    """Generate a unique audit event identifier."""
    return new_id("evt")


def new_user_id() -> str:
    """Generate a unique user identifier."""
    return new_id("usr")

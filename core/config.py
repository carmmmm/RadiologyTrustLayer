"""Central configuration loaded from environment variables / .env file."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

# ── Model ──────────────────────────────────────────────────────────────────
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
MEDGEMMA_MODEL_ID: str = os.getenv("MEDGEMMA_MODEL_ID", "google/medgemma-4b-it")
RTL_LORA_ID: str = os.getenv("RTL_LORA_ID", "")
MEDGEMMA_MOCK: bool = os.getenv("MEDGEMMA_MOCK", "false").lower() == "true"
MEDGEMMA_INFERENCE_MODE: str = os.getenv("MEDGEMMA_INFERENCE_MODE", "local")
MEDGEMMA_MAX_NEW_TOKENS: int = int(os.getenv("MEDGEMMA_MAX_NEW_TOKENS", "1024"))

# ── Pipeline ───────────────────────────────────────────────────────────────
RTL_PROMPT_VERSION: str = os.getenv("RTL_PROMPT_VERSION", "v1")
PROMPTS_DIR: Path = ROOT / "core" / "pipeline" / "prompts" / RTL_PROMPT_VERSION
SCHEMAS_DIR: Path = ROOT / "core" / "pipeline" / "schemas"

# ── Storage ────────────────────────────────────────────────────────────────
_storage_env = os.getenv("RTL_STORAGE_DIR", "")
STORAGE_DIR: Path = Path(_storage_env) if _storage_env else ROOT / "spaces_app" / "storage"

_db_env = os.getenv("RTL_DB_PATH", "")
DB_PATH: Path = Path(_db_env) if _db_env else STORAGE_DIR / "rtl.db"

RUNS_DIR: Path = STORAGE_DIR / "outputs" / "runs"
BATCHES_DIR: Path = STORAGE_DIR / "outputs" / "batches"

# ── Examples ────────────────────────────────────────────────────────────────
EXAMPLES_DIR: Path = ROOT / "spaces_app" / "ui" / "data" / "examples"
MOCK_RESULTS_PATH: Path = ROOT / "eval" / "sample_outputs" / "mock_results.json"

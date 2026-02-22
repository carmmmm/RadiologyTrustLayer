CREATE TABLE IF NOT EXISTS users (
  user_id TEXT PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  display_name TEXT NOT NULL,
  password_hash TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  created_at TEXT NOT NULL,

  input_image_hash TEXT NOT NULL,
  input_report_hash TEXT NOT NULL,
  case_label TEXT,

  model_name TEXT NOT NULL,
  model_version TEXT NOT NULL,
  lora_id TEXT,
  prompt_version TEXT NOT NULL,

  overall_score INTEGER NOT NULL,
  severity TEXT NOT NULL,

  flag_counts_json TEXT NOT NULL,
  status TEXT NOT NULL,
  error_message TEXT,

  results_path TEXT NOT NULL,

  FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS batches (
  batch_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  created_at TEXT NOT NULL,

  zip_name TEXT NOT NULL,
  num_cases_total INTEGER NOT NULL,
  num_cases_done INTEGER NOT NULL,
  num_cases_failed INTEGER NOT NULL,

  batch_summary_json TEXT NOT NULL,
  status TEXT NOT NULL,

  FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS batch_runs (
  batch_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  case_id TEXT NOT NULL,
  PRIMARY KEY (batch_id, run_id),
  FOREIGN KEY (batch_id) REFERENCES batches(batch_id),
  FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS audit_events (
  event_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  actor TEXT NOT NULL,
  event_type TEXT NOT NULL,
  details_json TEXT NOT NULL,
  FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

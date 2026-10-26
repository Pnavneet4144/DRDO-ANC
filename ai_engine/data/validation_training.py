import csv
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "master_training.csv"
EXPECTED_COLUMNS = [
    "id", "speaker_id", "clean_path", "noise_path",
    "reference_path", "noise_type", "snr_db", "split", "mixed_path",
]

with MANIFEST.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    assert reader.fieldnames == EXPECTED_COLUMNS, (
        f"Unexpected columns: {reader.fieldnames}"
    )
    rows = list(reader)

assert len(rows) == 2703, f"Expected 2703 rows, got {len(rows)}"
assert len({r["id"] for r in rows}) == len(rows), "Duplicate IDs found"
assert all(r["split"] == "train" for r in rows), "Unexpected split labels"

errors = []
for row in rows:
    for column in ("clean_path", "noise_path", "reference_path", "mixed_path"):
        path = ROOT / row[column]
        if not path.is_file():
            errors.append(f"{row['id']}: missing {column}: {path}")
            continue

        try:
            audio, sr = sf.read(path, dtype="float32")
            if sr != 16000:
                errors.append(f"{row['id']}: {column} has sample rate {sr}")
            if audio.ndim != 1:
                errors.append(f"{row['id']}: {column} is not mono")
            if len(audio) != 64000:
                errors.append(f"{row['id']}: {column} is not 4 seconds")
            if not np.isfinite(audio).all():
                errors.append(f"{row['id']}: {column} contains invalid values")
            if audio.size and np.max(np.abs(audio)) > 1.0:
                errors.append(f"{row['id']}: {column} exceeds [-1, 1]")
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            errors.append(f"{row['id']}: cannot read {column}: {exc}")

if errors:
    print(f"VALIDATION FAILED: {len(errors)} issue(s)")
    for error in errors[:30]:
        print("-", error)
    raise SystemExit(1)

print("VALIDATION PASSED")
print(f"Rows: {len(rows)}")
print("All four audio files per row exist and pass basic format checks.")
from __future__ import annotations

import re
from pathlib import Path
from typing import BinaryIO

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
NUMERIC_COLUMNS = ["age", "studytime", "failures", "absences", "G1", "G2", "G3"]
MODEL_COLUMNS = ["course", "age", "studytime", "failures", "absences", "G1", "G2", "G3"]

COURSES = [
    ("MATH", "Mathematics", "student-mat.csv"),
    ("POR", "Portuguese", "student-por.csv"),
]

DISPLAY_COLUMNS = ["record_id", "course", "school", *NUMERIC_COLUMNS]


def _validate_records(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    missing = [column for column in MODEL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}.")
    if frame.empty:
        raise ValueError("The dataset contains no records.")

    clean = frame.copy()
    for column in NUMERIC_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    if clean[NUMERIC_COLUMNS].isna().any().any():
        bad_columns = clean[NUMERIC_COLUMNS].columns[clean[NUMERIC_COLUMNS].isna().any()].tolist()
        raise ValueError(f"Missing or non-numeric values in: {', '.join(bad_columns)}.")
    if not clean["G1"].between(0, 20).all() or not clean["G2"].between(0, 20).all() or not clean["G3"].between(0, 20).all():
        raise ValueError("G1, G2, and G3 must use the 0–20 grade scale.")
    if (clean["absences"] < 0).any():
        raise ValueError("Absence counts cannot be negative.")
    if (clean["age"] < 0).any() or (clean["studytime"] < 1).any() or (clean["failures"] < 0).any():
        raise ValueError("Age, study time, and previous failure values are outside their valid ranges.")

    clean["course"] = clean["course"].astype(str).str.strip().replace({"": "Unspecified"})
    if "school" not in clean.columns:
        clean["school"] = source
    clean["school"] = clean["school"].fillna(source).astype(str)
    if "record_id" not in clean.columns:
        prefix = "UPLOAD" if source == "Uploaded CSV" else "RECORD"
        clean["record_id"] = [f"{prefix}-{index:04d}" for index in range(1, len(clean) + 1)]
    else:
        clean["record_id"] = clean["record_id"].fillna("").astype(str).str.strip()
        blank_ids = clean["record_id"].eq("")
        clean.loc[blank_ids, "record_id"] = [
            f"UPLOAD-{index:04d}" for index in range(1, int(blank_ids.sum()) + 1)
        ]

    clean["actual_outcome"] = clean["G3"].ge(10).map({True: "Pass", False: "Fail"})
    return clean[DISPLAY_COLUMNS + ["actual_outcome"]].reset_index(drop=True)


def load_uci_records(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Load and combine the two authentic UCI course files as separate course records."""
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    frames: list[pd.DataFrame] = []
    for prefix, course_name, filename in COURSES:
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"Required UCI data file was not found: {path}")
        raw = pd.read_csv(path, sep=";")
        required = {"school", *NUMERIC_COLUMNS}
        missing = sorted(required.difference(raw.columns))
        if missing:
            raise ValueError(f"{filename} is missing source columns: {', '.join(missing)}.")
        frame = raw[["school", *NUMERIC_COLUMNS]].copy()
        frame["course"] = course_name
        frame["record_id"] = [f"{prefix}-{index:04d}" for index in range(1, len(frame) + 1)]
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    return _validate_records(combined, "UCI")


def _canonical_column(name: str) -> str | None:
    token = re.sub(r"[^a-z0-9]", "", name.casefold())
    aliases = {
        "recordid": "record_id",
        "studentid": "record_id",
        "id": "record_id",
        "course": "course",
        "subject": "course",
        "school": "school",
        "age": "age",
        "studytime": "studytime",
        "failures": "failures",
        "absences": "absences",
        "g1": "G1",
        "g2": "G2",
        "g3": "G3",
    }
    return aliases.get(token)


def load_uploaded_records(uploaded_file: BinaryIO) -> pd.DataFrame:
    """Read a user-provided labeled CSV using the documented student-course schema."""
    try:
        raw = pd.read_csv(uploaded_file)
    except pd.errors.ParserError:
        uploaded_file.seek(0)
        raw = pd.read_csv(uploaded_file, sep=";")
    if len(raw.columns) == 1:
        uploaded_file.seek(0)
        raw = pd.read_csv(uploaded_file, sep=";")

    rename: dict[str, str] = {}
    for column in raw.columns:
        canonical = _canonical_column(str(column).strip())
        if canonical:
            rename[column] = canonical
    clean = raw.rename(columns=rename)
    required = set(MODEL_COLUMNS)
    missing = sorted(required.difference(clean.columns))
    if missing:
        raise ValueError(
            "Required CSV columns: course, age, studytime, failures, absences, G1, G2, and G3. "
            f"Missing: {', '.join(missing)}."
        )
    return _validate_records(clean, "Uploaded CSV")

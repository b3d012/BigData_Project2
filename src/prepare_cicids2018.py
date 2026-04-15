from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


# =========================================================
# Paths
# =========================================================
TEST_DIR = Path("dataset/test1")
PROCESSED_DIR = Path("dataset/processed")

FEATURES_JSON = PROCESSED_DIR / "feature_columns_binary.json"
LABEL_MAP_JSON = PROCESSED_DIR / "label_mapping_binary.json"

OUT_CSV = TEST_DIR / "ids2018_binary_merged.csv"
OUT_SUMMARY = TEST_DIR / "ids2018_binary_merged_summary.json"

CHUNK_SIZE = 200_000


# =========================================================
# Helpers
# =========================================================
def sanitize_column_name(name: str) -> str:
    name = str(name).strip()
    name = name.replace("/s", "_per_s")
    name = name.replace("/", "_")
    name = name.replace("-", "_")
    name = name.replace(".", "_")
    name = name.replace("(", "")
    name = name.replace(")", "")
    name = name.replace("%", "pct")
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name.lower()


def make_unique(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    out: list[str] = []

    for name in names:
        if name not in counts:
            counts[name] = 0
            out.append(name)
        else:
            counts[name] += 1
            out.append(f"{name}_{counts[name]}")
    return out


def normalize_label(label: str) -> str:
    label = str(label).strip()
    label = re.sub(r"\s+", " ", label)
    return label


def to_binary_label(label: str) -> str:
    label = normalize_label(label)
    if label.upper() == "BENIGN":
        return "BENIGN"
    return "ATTACK"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_usecols_for_file(path: Path, needed_sanitized: set[str]) -> list[str]:
    header_df = pd.read_csv(path, nrows=0, dtype=str, engine="python", encoding_errors="ignore")
    raw_cols = list(header_df.columns)

    keep_raw_cols = []
    for raw in raw_cols:
        san = sanitize_column_name(raw)
        if san in needed_sanitized:
            keep_raw_cols.append(raw)

    return keep_raw_cols


def stream_one_file(
    path: Path,
    needed_feature_cols: list[str],
    label_map: dict[str, int],
    header_written: bool,
    raw_label_counter: Counter,
    binary_label_counter: Counter,
) -> tuple[bool, int, int]:
    needed_sanitized = set(needed_feature_cols) | {"label"}
    usecols = get_usecols_for_file(path, needed_sanitized)

    if not usecols:
        raise ValueError(f"No usable columns found in {path.name}")

    rows_written = 0
    chunks_written = 0

    print(f"[LOAD] {path.name}")
    print(f"[INFO] Using {len(usecols)} raw columns from this file")

    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        chunksize=100_000,
        dtype=str,
        engine="python",
        on_bad_lines="skip",
        encoding_errors="ignore",
    ):
        clean_cols = [sanitize_column_name(c) for c in chunk.columns]
        clean_cols = make_unique(clean_cols)
        chunk.columns = clean_cols

        if "label" not in chunk.columns:
            raise ValueError(f"'label' column not found after sanitizing in {path.name}")

        chunk = chunk[chunk["label"].astype(str).str.strip().str.lower() != "label"].copy()

        if chunk.empty:
            continue

        chunk["label"] = chunk["label"].astype(str).map(normalize_label)
        chunk["target_name"] = chunk["label"].map(to_binary_label)
        chunk["target_id"] = chunk["target_name"].map(label_map)
        chunk["source_file"] = path.name

        for col in needed_feature_cols:
            if col not in chunk.columns:
                chunk[col] = pd.NA

        out_cols = ["source_file", "label", "target_name", "target_id"] + needed_feature_cols
        chunk = chunk[out_cols]

        raw_label_counter.update(chunk["label"].value_counts().to_dict())
        binary_label_counter.update(chunk["target_name"].value_counts().to_dict())

        chunk.to_csv(OUT_CSV, mode="a", index=False, header=not header_written)
        header_written = True

        rows_written += len(chunk)
        chunks_written += 1

    return header_written, rows_written, chunks_written


def main() -> None:
    csv_paths = sorted(
        p for p in TEST_DIR.glob("*.csv")
        if p.name not in {OUT_CSV.name}
    )
    print("[INFO] Input CSV files:")
    for p in csv_paths:
        print(" -", p.name)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {TEST_DIR.resolve()}")

    feature_cols = load_json(FEATURES_JSON)["feature_columns"]
    label_map = load_json(LABEL_MAP_JSON)

    if OUT_CSV.exists():
        OUT_CSV.unlink()

    raw_label_counter = Counter()
    binary_label_counter = Counter()

    total_rows = 0
    total_chunks = 0
    header_written = False
    file_stats = []

    for path in csv_paths:
        header_written, rows_written, chunks_written = stream_one_file(
            path=path,
            needed_feature_cols=feature_cols,
            label_map=label_map,
            header_written=header_written,
            raw_label_counter=raw_label_counter,
            binary_label_counter=binary_label_counter,
        )

        total_rows += rows_written
        total_chunks += chunks_written
        file_stats.append(
            {
                "file": path.name,
                "rows_written": int(rows_written),
                "chunks_written": int(chunks_written),
            }
        )

    summary = {
        "input_files": [p.name for p in csv_paths],
        "output_csv": str(OUT_CSV),
        "rows_written": int(total_rows),
        "chunks_written_total": int(total_chunks),
        "chunk_size": int(CHUNK_SIZE),
        "num_model_features_kept": len(feature_cols),
        "label_counts_raw": dict(raw_label_counter),
        "label_counts_binary": dict(binary_label_counter),
        "per_file_stats": file_stats,
    }

    save_json(OUT_SUMMARY, summary)

    print(f"[SAVE] {OUT_CSV}")
    print(f"[SAVE] {OUT_SUMMARY}")
    print("[DONE] IDS2018 streaming prep complete.")


if __name__ == "__main__":
    main()
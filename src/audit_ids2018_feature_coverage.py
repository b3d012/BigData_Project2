from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_CSV = Path("dataset/test1/ids2018_binary_merged.csv")
FEATURES_JSON = Path("dataset/processed/feature_columns_binary.json")
OUT_JSON = Path("dataset/test1/ids2018_feature_coverage.json")

CHUNK_SIZE = 200_000


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    feature_cols = load_json(FEATURES_JSON)["feature_columns"]

    total_rows = 0
    non_null_counts = {col: 0 for col in feature_cols}

    bad_tokens = {
        "Infinity": np.nan,
        "INFINITY": np.nan,
        "inf": np.nan,
        "Inf": np.nan,
        "-Infinity": np.nan,
        "-inf": np.nan,
        "-Inf": np.nan,
        "NaN": np.nan,
        "nan": np.nan,
        "": np.nan,
        " ": np.nan,
    }

    for chunk in pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE, low_memory=True):
        total_rows += len(chunk)

        X = chunk[feature_cols].copy()

        for col in feature_cols:
            X[col] = X[col].replace(bad_tokens)
            X[col] = pd.to_numeric(X[col], errors="coerce")

        X = X.replace([np.inf, -np.inf], np.nan)

        counts = X.notna().sum()
        for col in feature_cols:
            non_null_counts[col] += int(counts[col])

    coverage = []
    for col in feature_cols:
        non_null = non_null_counts[col]
        pct = non_null / total_rows if total_rows else 0.0
        coverage.append({
            "feature": col,
            "non_null_count": non_null,
            "non_null_fraction": pct,
        })

    coverage.sort(key=lambda x: x["non_null_fraction"])

    summary = {
        "input_csv": str(INPUT_CSV),
        "total_rows": total_rows,
        "features_with_zero_real_values": [x["feature"] for x in coverage if x["non_null_count"] == 0],
        "features_with_under_1pct_real_values": [x["feature"] for x in coverage if x["non_null_fraction"] < 0.01],
        "features_with_under_50pct_real_values": [x["feature"] for x in coverage if x["non_null_fraction"] < 0.50],
        "coverage": coverage,
    }

    save_json(OUT_JSON, summary)

    print("[DONE] Feature coverage audit complete.")
    print(f"[SAVE] {OUT_JSON}")
    print("Total rows:", total_rows)
    print("Zero-coverage features:", len(summary["features_with_zero_real_values"]))
    print("Under 1% coverage:", len(summary["features_with_under_1pct_real_values"]))
    print("Under 50% coverage:", len(summary["features_with_under_50pct_real_values"]))


if __name__ == "__main__":
    main()
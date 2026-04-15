from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# =========================================================
# Config
# =========================================================
RAW_DIR = Path("dataset/raw")
PROCESSED_DIR = Path("dataset/processed")

# "binary"  -> BENIGN vs ATTACK
# "multiclass" -> keeps attack families separated
LABEL_MODE = "binary"

RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Optional
DROP_EXACT_DUPLICATES = True
DROP_ALL_NAN_COLUMNS = True
DROP_CONSTANT_COLUMNS = True


# =========================================================
# Helpers
# =========================================================
def sanitize_column_name(name: str) -> str:
    """
    Convert raw CIC column names into clean snake_case names.
    Example:
        ' Flow Bytes/s' -> 'flow_bytes_per_s'
    """
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


def make_unique(names: List[str]) -> List[str]:
    """
    Ensure duplicate column names stay unique after sanitizing.
    """
    counts: Dict[str, int] = {}
    out: List[str] = []

    for name in names:
        if name not in counts:
            counts[name] = 0
            out.append(name)
        else:
            counts[name] += 1
            out.append(f"{name}_{counts[name]}")
    return out


def normalize_label(label: str) -> str:
    """
    Clean raw CIC labels into a consistent format.
    """
    label = str(label).strip()
    label = re.sub(r"\s+", " ", label)
    return label


def to_binary_label(label: str) -> str:
    """
    BENIGN stays BENIGN. Everything else becomes ATTACK.
    """
    label = normalize_label(label)
    if label.upper() == "BENIGN":
        return "BENIGN"
    return "ATTACK"


def to_multiclass_label(label: str) -> str:
    """
    Group CIC attack labels into cleaner families.
    """
    raw = normalize_label(label)
    upper = raw.upper()

    if upper == "BENIGN":
        return "BENIGN"

    if "DDoS".upper() in upper:
        return "DDOS"

    if "PORTSCAN" in upper:
        return "PORTSCAN"

    if "DOS HULK" in upper:
        return "DOS_HULK"
    if "DOS GOLDENEYE" in upper:
        return "DOS_GOLDENEYE"
    if "DOS SLOWHTTPTEST" in upper:
        return "DOS_SLOWHTTPTEST"
    if "DOS SLOWLORIS" in upper:
        return "DOS_SLOWLORIS"

    if "FTP-PATATOR" in upper:
        return "FTP_BRUTEFORCE"
    if "SSH-PATATOR" in upper:
        return "SSH_BRUTEFORCE"

    if "WEB ATTACK" in upper or "XSS" in upper or "SQL INJECTION" in upper or "BRUTE FORCE" in upper:
        return "WEB_ATTACK"

    if "INFILTRATION" in upper:
        return "INFILTRATION"

    if "BOT" in upper:
        return "BOT"

    if "HEARTBLEED" in upper:
        return "HEARTBLEED"

    return raw.upper().replace(" ", "_")


def load_one_csv(path: Path) -> pd.DataFrame:
    """
    Load one CIC CSV, clean columns, remove repeated header rows,
    and attach source filename for tracking.
    """
    print(f"[LOAD] {path.name}")
    df = pd.read_csv(path, low_memory=False)

    original_cols = [str(c).strip() for c in df.columns]
    clean_cols = [sanitize_column_name(c) for c in original_cols]
    clean_cols = make_unique(clean_cols)
    df.columns = clean_cols

    if "label" not in df.columns:
        raise ValueError(f"'label' column not found in {path.name}")

    # Remove repeated header rows sometimes embedded inside the file
    df = df[df["label"].astype(str).str.strip().str.lower() != "label"].copy()

    # Keep source filename for debugging / analysis
    df["source_file"] = path.name

    return df


def convert_features_to_numeric(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Convert all feature columns to numeric, coercing bad values to NaN.
    """
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

    for col in feature_cols:
        df[col] = df[col].replace(bad_tokens)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    return df


def build_target_column(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Create target_name and target_id.
    """
    if mode == "binary":
        df["target_name"] = df["label"].map(to_binary_label)
    elif mode == "multiclass":
        df["target_name"] = df["label"].map(to_multiclass_label)
    else:
        raise ValueError("LABEL_MODE must be 'binary' or 'multiclass'")

    unique_targets = sorted(df["target_name"].dropna().unique().tolist())
    target_to_id = {name: idx for idx, name in enumerate(unique_targets)}
    df["target_id"] = df["target_name"].map(target_to_id)

    return df, target_to_id


def drop_useless_columns(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """
    Drop all-NaN and constant columns.
    """
    dropped_all_nan: List[str] = []
    dropped_constant: List[str] = []

    if DROP_ALL_NAN_COLUMNS:
        dropped_all_nan = [c for c in feature_cols if df[c].isna().all()]
        if dropped_all_nan:
            df = df.drop(columns=dropped_all_nan)
            feature_cols = [c for c in feature_cols if c not in dropped_all_nan]

    if DROP_CONSTANT_COLUMNS:
        nunique = df[feature_cols].nunique(dropna=False)
        dropped_constant = nunique[nunique <= 1].index.tolist()
        if dropped_constant:
            df = df.drop(columns=dropped_constant)
            feature_cols = [c for c in feature_cols if c not in dropped_constant]

    return df, feature_cols, dropped_all_nan, dropped_constant


def median_impute(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Fill missing numeric values with median.
    """
    medians = df[feature_cols].median(numeric_only=True).to_dict()
    df[feature_cols] = df[feature_cols].fillna(medians)
    df[feature_cols] = df[feature_cols].fillna(0)
    return df, medians


def save_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(RAW_DIR.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR.resolve()}")

    print(f"[INFO] Found {len(csv_paths)} CSV files")

    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        df = load_one_csv(path)
        frames.append(df)

    merged = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    print(f"[INFO] Merged shape: {merged.shape}")

    # Clean label text
    merged["label"] = merged["label"].astype(str).map(normalize_label)

    # Remove empty labels if any
    merged = merged[merged["label"].notna()].copy()

    # Remove exact duplicate rows
    if DROP_EXACT_DUPLICATES:
        before = len(merged)
        merged = merged.drop_duplicates().copy()
        after = len(merged)
        print(f"[INFO] Dropped exact duplicates: {before - after}")

    # Keep source_file but do not use as a feature
    non_feature_cols = {"label", "source_file"}
    feature_cols = [c for c in merged.columns if c not in non_feature_cols]

    # Convert features to numeric
    merged = convert_features_to_numeric(merged, feature_cols)

    # Build target
    merged, target_to_id = build_target_column(merged, LABEL_MODE)

    # Reorder some columns for readability
    ordered_cols = ["source_file", "label", "target_name", "target_id"] + feature_cols
    merged = merged[ordered_cols]

    # Rebuild feature list after column ordering
    feature_cols = [c for c in merged.columns if c not in {"source_file", "label", "target_name", "target_id"}]

    # Drop useless columns
    merged, feature_cols, dropped_all_nan, dropped_constant = drop_useless_columns(merged, feature_cols)

    # Missing values before imputation
    missing_before = int(merged[feature_cols].isna().sum().sum())
    print(f"[INFO] Missing feature values before imputation: {missing_before}")

    # Impute
    merged, medians = median_impute(merged, feature_cols)

    missing_after = int(merged[feature_cols].isna().sum().sum())
    print(f"[INFO] Missing feature values after imputation: {missing_after}")

    # Final reorder
    final_cols = ["source_file", "label", "target_name", "target_id"] + feature_cols
    merged = merged[final_cols]

    # Save full cleaned dataset
    full_out = PROCESSED_DIR / f"cicids2017_{LABEL_MODE}_full.csv"
    merged.to_csv(full_out, index=False)
    print(f"[SAVE] {full_out}")

    # Stratified train / val / test split
    train_df, temp_df = train_test_split(
        merged,
        test_size=(1.0 - TRAIN_SIZE),
        random_state=RANDOM_STATE,
        stratify=merged["target_id"],
    )

    val_fraction_of_temp = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_fraction_of_temp),
        random_state=RANDOM_STATE,
        stratify=temp_df["target_id"],
    )

    train_out = PROCESSED_DIR / f"cicids2017_{LABEL_MODE}_train.csv"
    val_out = PROCESSED_DIR / f"cicids2017_{LABEL_MODE}_val.csv"
    test_out = PROCESSED_DIR / f"cicids2017_{LABEL_MODE}_test.csv"

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)

    print(f"[SAVE] {train_out}")
    print(f"[SAVE] {val_out}")
    print(f"[SAVE] {test_out}")

    # Save feature columns
    feature_cols_out = PROCESSED_DIR / f"feature_columns_{LABEL_MODE}.json"
    save_json(feature_cols_out, {"feature_columns": feature_cols})
    print(f"[SAVE] {feature_cols_out}")

    # Save label mapping
    label_map_out = PROCESSED_DIR / f"label_mapping_{LABEL_MODE}.json"
    save_json(label_map_out, target_to_id)
    print(f"[SAVE] {label_map_out}")

    # Save imputer values
    medians_out = PROCESSED_DIR / f"median_imputer_{LABEL_MODE}.json"
    save_json(medians_out, medians)
    print(f"[SAVE] {medians_out}")

    # Save summary
    summary = {
        "label_mode": LABEL_MODE,
        "random_state": RANDOM_STATE,
        "input_files": [p.name for p in csv_paths],
        "full_shape": {
            "rows": int(merged.shape[0]),
            "cols": int(merged.shape[1]),
        },
        "num_features": len(feature_cols),
        "dropped_all_nan_columns": dropped_all_nan,
        "dropped_constant_columns": dropped_constant,
        "missing_before_imputation": missing_before,
        "missing_after_imputation": missing_after,
        "class_distribution_full": merged["target_name"].value_counts().to_dict(),
        "class_distribution_train": train_df["target_name"].value_counts().to_dict(),
        "class_distribution_val": val_df["target_name"].value_counts().to_dict(),
        "class_distribution_test": test_df["target_name"].value_counts().to_dict(),
    }

    summary_out = PROCESSED_DIR / f"preprocess_summary_{LABEL_MODE}.json"
    save_json(summary_out, summary)
    print(f"[SAVE] {summary_out}")

    print("\n[DONE] Preprocessing complete.")
    print(f"[INFO] Final feature count: {len(feature_cols)}")
    print(f"[INFO] Final classes: {sorted(target_to_id.keys())}")


if __name__ == "__main__":
    main()
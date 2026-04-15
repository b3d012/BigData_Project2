from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROCESSED_DIR = Path("dataset/processed")
MODEL_DIR = Path("models/baseline_xgb")
OUTPUT_DIR = Path("outputs/inference")

FEATURES_JSON = PROCESSED_DIR / "feature_columns_binary.json"
MEDIANS_JSON = PROCESSED_DIR / "median_imputer_binary.json"
LABEL_MAP_JSON = PROCESSED_DIR / "label_mapping_binary.json"

MODEL_PATH = MODEL_DIR / "xgb_baseline_model.joblib"
THRESHOLD_PATH = MODEL_DIR / "selected_threshold.json"

CHUNK_SIZE = 200_000


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(data), f, indent=2)


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


def align_features(df: pd.DataFrame, feature_cols: list[str], medians: dict[str, float]):
    missing_features = [c for c in feature_cols if c not in df.columns]
    extra_features = [c for c in df.columns if c not in feature_cols]

    for col in missing_features:
        df[col] = np.nan

    X = df[feature_cols].copy()

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
        X[col] = X[col].replace(bad_tokens)
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(medians)
    X = X.fillna(0)

    return X, missing_features, extra_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunked XGBoost inference on a large CSV")
    parser.add_argument("--input", required=True, help="Path to merged IDS2018 CSV")
    parser.add_argument("--name", default="ids2018_stream", help="Run name")
    args = parser.parse_args()

    input_path = Path(args.input)
    run_name = args.name

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = load_json(FEATURES_JSON)["feature_columns"]
    medians = load_json(MEDIANS_JSON)
    label_map = load_json(LABEL_MAP_JSON)
    threshold_info = load_json(THRESHOLD_PATH)

    attack_class_id = int(label_map["ATTACK"])
    benign_class_id = int(label_map["BENIGN"])
    attack_threshold = float(threshold_info["best_threshold"])

    model = joblib.load(MODEL_PATH)

    predictions_out = OUTPUT_DIR / f"{run_name}_predictions.csv"
    summary_out = OUTPUT_DIR / f"{run_name}_summary.json"

    if predictions_out.exists():
        predictions_out.unlink()

    total_rows = 0
    total_attack_pred = 0
    total_benign_pred = 0
    total_attack_prob_sum = 0.0
    total_chunks = 0

    missing_feature_union = set()
    extra_feature_union = set()
    header_written = False

    for chunk in pd.read_csv(input_path, chunksize=CHUNK_SIZE, low_memory=True):
        clean_cols = [sanitize_column_name(c) for c in chunk.columns]
        clean_cols = make_unique(clean_cols)
        chunk.columns = clean_cols

        if "label" in chunk.columns:
            chunk = chunk[chunk["label"].astype(str).str.strip().str.lower() != "label"].copy()

        if chunk.empty:
            continue

        X, missing_features, extra_features = align_features(chunk, feature_cols, medians)

        missing_feature_union.update(missing_features)
        extra_feature_union.update(extra_features)

        proba = model.predict_proba(X)
        classes = [int(x) for x in model.classes_]
        attack_idx = classes.index(attack_class_id)
        benign_idx = classes.index(benign_class_id)

        attack_probs = proba[:, attack_idx]
        benign_probs = proba[:, benign_idx]

        y_pred = np.where(attack_probs >= attack_threshold, attack_class_id, benign_class_id)
        pred_name = np.where(y_pred == attack_class_id, "ATTACK", "BENIGN")

        out_chunk = pd.DataFrame({
            "attack_probability": attack_probs,
            "benign_probability": benign_probs,
            "predicted_target_id": y_pred,
            "predicted_target_name": pred_name,
            "used_attack_threshold": attack_threshold,
        })

        if "label" in chunk.columns:
            out_chunk.insert(0, "label", chunk["label"].astype(str).values)
        if "target_name" in chunk.columns:
            out_chunk.insert(1 if "label" in chunk.columns else 0, "target_name", chunk["target_name"].astype(str).values)
        if "target_id" in chunk.columns:
            insert_pos = 2 if "label" in chunk.columns and "target_name" in chunk.columns else len(out_chunk.columns)
            out_chunk.insert(insert_pos, "target_id", pd.to_numeric(chunk["target_id"], errors="coerce").values)

        out_chunk.to_csv(predictions_out, mode="a", index=False, header=not header_written)
        header_written = True

        total_rows += len(out_chunk)
        total_attack_pred += int((y_pred == attack_class_id).sum())
        total_benign_pred += int((y_pred == benign_class_id).sum())
        total_attack_prob_sum += float(np.sum(attack_probs))
        total_chunks += 1

        print(f"[CHUNK {total_chunks}] rows={len(out_chunk)} total_rows={total_rows}")

    summary = {
        "input_csv": str(input_path),
        "num_rows": total_rows,
        "chunks_processed": total_chunks,
        "attack_threshold": attack_threshold,
        "predicted_attack_count": total_attack_pred,
        "predicted_benign_count": total_benign_pred,
        "predicted_attack_rate": total_attack_pred / total_rows if total_rows else 0.0,
        "average_attack_probability": total_attack_prob_sum / total_rows if total_rows else 0.0,
        "missing_feature_count": len(missing_feature_union),
        "missing_features": sorted(missing_feature_union),
        "extra_column_count": len(extra_feature_union),
        "extra_columns": sorted(extra_feature_union),
        "classes_seen_by_model": [attack_class_id, benign_class_id],
        "predictions_csv": str(predictions_out),
    }

    save_json(summary_out, summary)

    print("\n[DONE] Streaming inference complete.")
    print(f"[SAVE] {predictions_out}")
    print(f"[SAVE] {summary_out}")
    print("[SUMMARY]")
    print("Rows:", total_rows)
    print("Predicted ATTACK:", total_attack_pred)
    print("Predicted BENIGN:", total_benign_pred)
    print("Predicted ATTACK rate:", round(summary["predicted_attack_rate"], 6))
    print("Average ATTACK probability:", round(summary["average_attack_probability"], 6))
    print("Missing features:", len(missing_feature_union))
    print("Extra columns:", len(extra_feature_union))


if __name__ == "__main__":
    main()
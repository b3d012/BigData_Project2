from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =========================================================
# Paths
# =========================================================
PROCESSED_DIR = Path("dataset/processed")
MODEL_DIR = Path("models/baseline_xgb")
OUTPUT_DIR = Path("outputs/inference")

FEATURES_JSON = PROCESSED_DIR / "feature_columns_binary.json"
MEDIANS_JSON = PROCESSED_DIR / "median_imputer_binary.json"
LABEL_MAP_JSON = PROCESSED_DIR / "label_mapping_binary.json"

MODEL_PATH = MODEL_DIR / "xgb_baseline_model.joblib"
THRESHOLD_PATH = MODEL_DIR / "selected_threshold.json"


# =========================================================
# Helpers
# =========================================================
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


def load_input_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)

    clean_cols = [sanitize_column_name(c) for c in df.columns]
    clean_cols = make_unique(clean_cols)
    df.columns = clean_cols

    if "label" in df.columns:
        df = df[df["label"].astype(str).str.strip().str.lower() != "label"].copy()

    return df


def align_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    medians: dict[str, float],
) -> tuple[pd.DataFrame, dict]:
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

    info = {
        "missing_feature_count": len(missing_features),
        "missing_features": missing_features,
        "extra_column_count": len(extra_features),
        "extra_columns": extra_features,
    }

    return X, info


def extract_ground_truth(df: pd.DataFrame, label_map: dict[str, int]) -> np.ndarray | None:
    if "target_id" in df.columns:
        return pd.to_numeric(df["target_id"], errors="coerce").fillna(-1).astype(int).to_numpy()

    if "target_name" in df.columns:
        target_name = df["target_name"].astype(str).str.strip().str.upper()
        mapping_upper = {k.upper(): v for k, v in label_map.items()}
        y = target_name.map(mapping_upper)
        if y.notna().all():
            return y.astype(int).to_numpy()

    if "label" in df.columns:
        target_name = df["label"].map(to_binary_label)
        y = target_name.map(label_map)
        if y.notna().all():
            return y.astype(int).to_numpy()

    return None


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attack_probs: np.ndarray,
    attack_class_id: int,
    benign_class_id: int,
    threshold: float,
) -> dict:
    y_attack_true = (y_true == attack_class_id).astype(int)
    y_attack_pred = (y_pred == attack_class_id).astype(int)

    return {
        "attack_threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_attack": float(precision_score(y_attack_true, y_attack_pred, zero_division=0)),
        "recall_attack": float(recall_score(y_attack_true, y_attack_pred, zero_division=0)),
        "f1_attack": float(f1_score(y_attack_true, y_attack_pred, zero_division=0)),
        "roc_auc_attack": float(roc_auc_score(y_attack_true, attack_probs)),
        "average_precision_attack": float(average_precision_score(y_attack_true, attack_probs)),
        "confusion_matrix_labels_[attack,benign]": confusion_matrix(
            y_true, y_pred, labels=[attack_class_id, benign_class_id]
        ).tolist(),
        "classification_report_attack_positive": classification_report(
            y_attack_true,
            y_attack_pred,
            target_names=["BENIGN", "ATTACK"],
            output_dict=True,
            zero_division=0,
        ),
    }


# =========================================================
# Main
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Run XGBoost CIC-IDS2017 inference on a CSV file.")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--name", default=None, help="Optional run name for outputs")
    args = parser.parse_args()

    input_path = Path(args.input)
    run_name = args.name if args.name else input_path.stem

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading saved artifacts...")
    feature_cols = load_json(FEATURES_JSON)["feature_columns"]
    medians = load_json(MEDIANS_JSON)
    label_map = load_json(LABEL_MAP_JSON)
    threshold_info = load_json(THRESHOLD_PATH)

    attack_class_id = int(label_map["ATTACK"])
    benign_class_id = int(label_map["BENIGN"])
    attack_threshold = float(threshold_info["best_threshold"])

    model = joblib.load(MODEL_PATH)

    print("[INFO] Loading input CSV...")
    df = load_input_csv(input_path)
    print(f"[INFO] Input shape before alignment: {df.shape}")

    y_true = extract_ground_truth(df, label_map)

    X, alignment_info = align_features(df, feature_cols, medians)
    print(f"[INFO] Aligned feature matrix shape: {X.shape}")
    print(f"[INFO] Missing expected features added with median fill: {alignment_info['missing_feature_count']}")

    print("[INFO] Running model inference...")
    proba = model.predict_proba(X)

    classes = [int(x) for x in model.classes_]
    attack_idx = classes.index(attack_class_id)
    benign_idx = classes.index(benign_class_id)

    attack_probs = proba[:, attack_idx]
    benign_probs = proba[:, benign_idx]

    y_pred = np.where(attack_probs >= attack_threshold, attack_class_id, benign_class_id)
    pred_name = np.where(y_pred == attack_class_id, "ATTACK", "BENIGN")

    result_df = df.copy()
    result_df["attack_probability"] = attack_probs
    result_df["benign_probability"] = benign_probs
    result_df["predicted_target_id"] = y_pred
    result_df["predicted_target_name"] = pred_name
    result_df["used_attack_threshold"] = attack_threshold

    predictions_out = OUTPUT_DIR / f"{run_name}_predictions.csv"
    result_df.to_csv(predictions_out, index=False)

    summary = {
        "input_csv": str(input_path),
        "num_rows": int(len(result_df)),
        "attack_threshold": attack_threshold,
        "predicted_attack_count": int((y_pred == attack_class_id).sum()),
        "predicted_benign_count": int((y_pred == benign_class_id).sum()),
        "predicted_attack_rate": float((y_pred == attack_class_id).mean()),
        "average_attack_probability": float(np.mean(attack_probs)),
        "alignment_info": alignment_info,
        "classes_seen_by_model": classes,
    }

    summary_out = OUTPUT_DIR / f"{run_name}_summary.json"
    save_json(summary_out, summary)

    print("\n[SUMMARY]")
    print("Rows:", len(result_df))
    print("Predicted ATTACK:", summary["predicted_attack_count"])
    print("Predicted BENIGN:", summary["predicted_benign_count"])
    print("Predicted ATTACK rate:", round(summary["predicted_attack_rate"], 6))
    print("Average ATTACK probability:", round(summary["average_attack_probability"], 6))
    print("Threshold used:", attack_threshold)

    if y_true is not None:
        print("\n[INFO] Ground-truth labels found. Computing evaluation metrics...")
        metrics = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            attack_probs=attack_probs,
            attack_class_id=attack_class_id,
            benign_class_id=benign_class_id,
            threshold=attack_threshold,
        )

        metrics_out = OUTPUT_DIR / f"{run_name}_metrics.json"
        save_json(metrics_out, metrics)

        print("Accuracy:", round(metrics["accuracy"], 6))
        print("Precision (ATTACK):", round(metrics["precision_attack"], 6))
        print("Recall (ATTACK):", round(metrics["recall_attack"], 6))
        print("F1 (ATTACK):", round(metrics["f1_attack"], 6))
        print("ROC-AUC (ATTACK):", round(metrics["roc_auc_attack"], 6))
        print("Average Precision (ATTACK):", round(metrics["average_precision_attack"], 6))
        print(f"[SAVE] {metrics_out}")

    print(f"\n[SAVE] {predictions_out}")
    print(f"[SAVE] {summary_out}")
    print("\n[DONE] Inference complete.")


if __name__ == "__main__":
    main()
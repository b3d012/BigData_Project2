from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from xgboost import XGBClassifier


# =========================================================
# Config
# =========================================================
PROCESSED_DIR = Path("dataset/processed")
MODEL_DIR = Path("models/baseline_xgb")

TRAIN_CSV = PROCESSED_DIR / "cicids2017_binary_train.csv"
VAL_CSV = PROCESSED_DIR / "cicids2017_binary_val.csv"
TEST_CSV = PROCESSED_DIR / "cicids2017_binary_test.csv"
FEATURES_JSON = PROCESSED_DIR / "feature_columns_binary.json"
LABEL_MAP_JSON = PROCESSED_DIR / "label_mapping_binary.json"

RANDOM_STATE = 42

# XGBoost params
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# =========================================================
# Helpers
# =========================================================
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_split(csv_path: Path, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path, low_memory=False)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected features in {csv_path.name}: {missing}")

    X = df[feature_cols].copy()
    y = df["target_id"].copy()

    return X, y


def compute_scale_pos_weight(y: pd.Series) -> float:
    """
    For binary labels where:
      0 = ATTACK
      1 = BENIGN
    XGBoost's scale_pos_weight applies to the positive class (label 1).
    Since ATTACK is the minority and BENIGN is label 1 in your mapping,
    we will instead use per-sample weights below for correct minority focus.

    This function is kept for reference but is not used.
    """
    num_pos = int((y == 1).sum())
    num_neg = int((y == 0).sum())
    if num_pos == 0:
        return 1.0
    return num_neg / num_pos


def build_sample_weights(y: pd.Series) -> np.ndarray:
    """
    Give higher weight to the minority class.
    """
    class_counts = y.value_counts().to_dict()
    n_samples = len(y)
    n_classes = len(class_counts)

    class_weights = {
        cls: n_samples / (n_classes * count)
        for cls, count in class_counts.items()
    }

    return y.map(class_weights).astype(float).to_numpy()


def choose_best_threshold(y_true: np.ndarray, y_prob_attack: np.ndarray) -> dict:
    """
    Pick threshold that maximizes F1 for ATTACK as positive class.
    target_id mapping from your preprocessing:
      ATTACK -> 0
      BENIGN -> 1

    So we convert to:
      y_attack = 1 if ATTACK else 0
    """
    y_attack = (y_true == 0).astype(int)

    precision, recall, thresholds = precision_recall_curve(y_attack, y_prob_attack)

    # thresholds has length n-1 relative to precision/recall
    f1_scores = []
    usable_thresholds = []

    for i, thr in enumerate(thresholds):
        p = precision[i]
        r = recall[i]
        if (p + r) == 0:
            f1 = 0.0
        else:
            f1 = 2 * p * r / (p + r)
        f1_scores.append(float(f1))
        usable_thresholds.append(float(thr))

    best_idx = int(np.argmax(f1_scores))
    return {
        "best_threshold": usable_thresholds[best_idx],
        "best_f1": f1_scores[best_idx],
        "precision_at_best_f1": float(precision[best_idx]),
        "recall_at_best_f1": float(recall[best_idx]),
    }


def evaluate_split(
    split_name: str,
    y_true: np.ndarray,
    y_prob_benign: np.ndarray,
    attack_threshold: float,
) -> dict:
    """
    Convert benign probability to attack probability, then threshold ATTACK.
    """
    y_prob_attack = 1.0 - y_prob_benign

    # Predict ATTACK when attack prob >= threshold
    # target_id: ATTACK = 0, BENIGN = 1
    y_pred = np.where(y_prob_attack >= attack_threshold, 0, 1)

    y_attack_true = (y_true == 0).astype(int)
    y_attack_pred = (y_pred == 0).astype(int)

    metrics = {
        "split": split_name,
        "attack_threshold": float(attack_threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_attack": float(precision_score(y_attack_true, y_attack_pred, zero_division=0)),
        "recall_attack": float(recall_score(y_attack_true, y_attack_pred, zero_division=0)),
        "f1_attack": float(f1_score(y_attack_true, y_attack_pred, zero_division=0)),
        "roc_auc_attack": float(roc_auc_score(y_attack_true, y_prob_attack)),
        "average_precision_attack": float(average_precision_score(y_attack_true, y_prob_attack)),
        "confusion_matrix_labels_[attack,benign]": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
        "classification_report_attack_positive": classification_report(
            y_attack_true,
            y_attack_pred,
            target_names=["BENIGN", "ATTACK"],
            output_dict=True,
            zero_division=0,
        ),
    }

    return metrics


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# =========================================================
# Main
# =========================================================
def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading feature schema...")
    feature_cols = load_json(FEATURES_JSON)["feature_columns"]
    label_map = load_json(LABEL_MAP_JSON)

    print("[INFO] Label mapping:", label_map)
    print(f"[INFO] Number of features: {len(feature_cols)}")

    print("[INFO] Loading train/val/test splits...")
    X_train, y_train = load_split(TRAIN_CSV, feature_cols)
    X_val, y_val = load_split(VAL_CSV, feature_cols)
    X_test, y_test = load_split(TEST_CSV, feature_cols)

    print(f"[INFO] Train shape: {X_train.shape}")
    print(f"[INFO] Val shape:   {X_val.shape}")
    print(f"[INFO] Test shape:  {X_test.shape}")

    train_class_counts = y_train.value_counts().sort_index().to_dict()
    print(f"[INFO] Train class counts: {train_class_counts}")

    sample_weights = build_sample_weights(y_train)

    model = XGBClassifier(**XGB_PARAMS)

    print("[INFO] Training XGBoost baseline...")
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True,
    )

    print("[INFO] Generating validation probabilities...")
    val_proba = model.predict_proba(X_val)
    val_prob_benign = val_proba[:, 1]
    val_prob_attack = 1.0 - val_prob_benign

    threshold_info = choose_best_threshold(y_val.to_numpy(), val_prob_attack)
    best_threshold = threshold_info["best_threshold"]

    print("[INFO] Best validation threshold for ATTACK:", best_threshold)
    print("[INFO] Validation F1 at best threshold:", threshold_info["best_f1"])

    print("[INFO] Evaluating validation split...")
    val_metrics = evaluate_split(
        split_name="val",
        y_true=y_val.to_numpy(),
        y_prob_benign=val_prob_benign,
        attack_threshold=best_threshold,
    )

    print("[INFO] Evaluating test split...")
    test_proba = model.predict_proba(X_test)
    test_prob_benign = test_proba[:, 1]

    test_metrics = evaluate_split(
        split_name="test",
        y_true=y_test.to_numpy(),
        y_prob_benign=test_prob_benign,
        attack_threshold=best_threshold,
    )

    print("[INFO] Saving model and artifacts...")
    model.save_model(str(MODEL_DIR / "xgb_baseline_model.json"))
    joblib.dump(model, MODEL_DIR / "xgb_baseline_model.joblib")

    save_json(MODEL_DIR / "xgb_params.json", XGB_PARAMS)
    save_json(MODEL_DIR / "selected_threshold.json", threshold_info)
    save_json(MODEL_DIR / "val_metrics.json", val_metrics)
    save_json(MODEL_DIR / "test_metrics.json", test_metrics)

    inference_schema = {
        "feature_columns": feature_cols,
        "label_mapping": label_map,
        "attack_class_id": 0,
        "benign_class_id": 1,
        "probability_column_meaning": {
            "predict_proba_col_0": "ATTACK",
            "predict_proba_col_1": "BENIGN",
        },
    }
    save_json(MODEL_DIR / "inference_schema.json", inference_schema)

    print("\n[DONE] XGBoost baseline training complete.")
    print(f"[SAVE] {MODEL_DIR / 'xgb_baseline_model.json'}")
    print(f"[SAVE] {MODEL_DIR / 'xgb_baseline_model.joblib'}")
    print(f"[SAVE] {MODEL_DIR / 'selected_threshold.json'}")
    print(f"[SAVE] {MODEL_DIR / 'val_metrics.json'}")
    print(f"[SAVE] {MODEL_DIR / 'test_metrics.json'}")
    print(f"[SAVE] {MODEL_DIR / 'inference_schema.json'}")

    print("\n[SUMMARY]")
    print("Validation F1 (ATTACK):", round(val_metrics["f1_attack"], 6))
    print("Validation Recall (ATTACK):", round(val_metrics["recall_attack"], 6))
    print("Validation Precision (ATTACK):", round(val_metrics["precision_attack"], 6))
    print("Validation ROC-AUC (ATTACK):", round(val_metrics["roc_auc_attack"], 6))
    print("Validation AP (ATTACK):", round(val_metrics["average_precision_attack"], 6))

    print("Test F1 (ATTACK):", round(test_metrics["f1_attack"], 6))
    print("Test Recall (ATTACK):", round(test_metrics["recall_attack"], 6))
    print("Test Precision (ATTACK):", round(test_metrics["precision_attack"], 6))
    print("Test ROC-AUC (ATTACK):", round(test_metrics["roc_auc_attack"], 6))
    print("Test AP (ATTACK):", round(test_metrics["average_precision_attack"], 6))


if __name__ == "__main__":
    main()
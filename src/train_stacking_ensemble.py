from __future__ import annotations

import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


# =========================================================
# Config
# =========================================================
PROCESSED_DIR = Path("dataset/processed")
MODEL_DIR = Path("models/stacking_ensemble")

TRAIN_CSV = PROCESSED_DIR / "cicids2017_binary_train.csv"
VAL_CSV = PROCESSED_DIR / "cicids2017_binary_val.csv"
TEST_CSV = PROCESSED_DIR / "cicids2017_binary_test.csv"
FEATURES_JSON = PROCESSED_DIR / "feature_columns_binary.json"
LABEL_MAP_JSON = PROCESSED_DIR / "label_mapping_binary.json"

RANDOM_STATE = 42
N_SPLITS = 3

BASE_MODEL_ORDER = ["rf", "hgb", "lr", "xgb"]
META_MODEL_NAME = "meta_logreg"

RF_PARAMS = {
    "n_estimators": 180,
    "max_depth": 18,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": True,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

HGB_PARAMS = {
    "loss": "log_loss",
    "learning_rate": 0.08,
    "max_iter": 220,
    "max_depth": 12,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "early_stopping": False,
    "random_state": RANDOM_STATE,
}

LR_PARAMS = {
    "solver": "saga",
    "l1_ratio": 0.0,
    "C": 1.0,
    "max_iter": 300,
    "tol": 1e-3,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

XGB_PARAMS = {
    "n_estimators": 250,
    "max_depth": 8,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

META_LR_PARAMS = {
    "solver": "lbfgs",
    "l1_ratio": 0.0,
    "C": 1.0,
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}


# =========================================================
# Helpers
# =========================================================
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_split(csv_path: Path, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, low_memory=False)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected features in {csv_path.name}: {missing}")

    X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y = df["target_id"].to_numpy(dtype=np.int32, copy=True)

    X = np.ascontiguousarray(X)
    y = np.ascontiguousarray(y)

    return X, y


def build_model(model_name: str):
    if model_name == "rf":
        return RandomForestClassifier(**RF_PARAMS)
    if model_name == "hgb":
        return HistGradientBoostingClassifier(**HGB_PARAMS)
    if model_name == "lr":
        return LogisticRegression(**LR_PARAMS)
    if model_name == "xgb":
        return XGBClassifier(**XGB_PARAMS)
    if model_name == META_MODEL_NAME:
        return LogisticRegression(**META_LR_PARAMS)
    raise ValueError(f"Unknown model_name: {model_name}")


def fit_model(model_name: str, model, X: np.ndarray, y: np.ndarray):
    """
    Fit with balanced sample weights where needed.
    """
    if model_name in {"hgb", "xgb"}:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        model.fit(X, y, sample_weight=sample_weight)
    else:
        model.fit(X, y)
    return model


def get_attack_proba(model, X: np.ndarray, attack_class_id: int) -> np.ndarray:
    """
    Return probability of ATTACK class regardless of class order.
    """
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if attack_class_id not in classes:
        raise ValueError(f"Attack class {attack_class_id} not found in model.classes_: {classes}")
    attack_idx = classes.index(attack_class_id)
    return proba[:, attack_idx].astype(np.float32)


def choose_best_threshold(y_true: np.ndarray, attack_probs: np.ndarray, attack_class_id: int) -> dict:
    """
    Tune ATTACK threshold on validation scores by maximizing F1.
    """
    y_attack = (y_true == attack_class_id).astype(int)

    precision, recall, thresholds = precision_recall_curve(y_attack, attack_probs)

    if len(thresholds) == 0:
        return {
            "best_threshold": 0.5,
            "best_f1": 0.0,
            "precision_at_best_f1": 0.0,
            "recall_at_best_f1": 0.0,
        }

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


def evaluate_with_probs(
    split_name: str,
    y_true: np.ndarray,
    attack_probs: np.ndarray,
    attack_threshold: float,
    attack_class_id: int,
    benign_class_id: int,
) -> dict:
    y_pred = np.where(attack_probs >= attack_threshold, attack_class_id, benign_class_id)

    y_attack_true = (y_true == attack_class_id).astype(int)
    y_attack_pred = (y_pred == attack_class_id).astype(int)

    return {
        "split": split_name,
        "attack_threshold": float(attack_threshold),
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
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = load_json(FEATURES_JSON)["feature_columns"]
    label_map = load_json(LABEL_MAP_JSON)

    if "ATTACK" not in label_map or "BENIGN" not in label_map:
        raise ValueError(f"Expected ATTACK and BENIGN in label mapping, got: {label_map}")

    attack_class_id = int(label_map["ATTACK"])
    benign_class_id = int(label_map["BENIGN"])

    print("[INFO] Loading data...")
    X_train, y_train = load_split(TRAIN_CSV, feature_cols)
    X_val, y_val = load_split(VAL_CSV, feature_cols)
    X_test, y_test = load_split(TEST_CSV, feature_cols)

    print(f"[INFO] Train shape: {X_train.shape}")
    print(f"[INFO] Val shape:   {X_val.shape}")
    print(f"[INFO] Test shape:  {X_test.shape}")
    print(f"[INFO] Label mapping: {label_map}")

    meta_feature_names = [f"{name}_prob_attack" for name in BASE_MODEL_ORDER]

    # -----------------------------------------------------
    # Step 1: OOF predictions on training split
    # -----------------------------------------------------
    print("\n[STEP 1] Building out-of-fold predictions for the meta-learner...")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_meta_X = np.zeros((len(X_train), len(BASE_MODEL_ORDER)), dtype=np.float32)
    fold_summary = []

    for fold_num, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
        print(f"\n[FOLD {fold_num}/{N_SPLITS}]")

        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_va, y_va = X_train[va_idx], y_train[va_idx]

        fold_info = {"fold": fold_num, "models": {}}

        for model_idx, model_name in enumerate(BASE_MODEL_ORDER):
            print(f"[FOLD {fold_num}] Training {model_name}...")
            model = build_model(model_name)
            model = fit_model(model_name, model, X_tr, y_tr)

            va_attack_probs = get_attack_proba(model, X_va, attack_class_id)
            oof_meta_X[va_idx, model_idx] = va_attack_probs

            # Quick fold-level metric at threshold 0.5 for monitoring
            fold_metrics = evaluate_with_probs(
                split_name=f"fold_{fold_num}_valid",
                y_true=y_va,
                attack_probs=va_attack_probs,
                attack_threshold=0.5,
                attack_class_id=attack_class_id,
                benign_class_id=benign_class_id,
            )
            fold_info["models"][model_name] = {
                "accuracy": fold_metrics["accuracy"],
                "precision_attack": fold_metrics["precision_attack"],
                "recall_attack": fold_metrics["recall_attack"],
                "f1_attack": fold_metrics["f1_attack"],
                "roc_auc_attack": fold_metrics["roc_auc_attack"],
                "average_precision_attack": fold_metrics["average_precision_attack"],
            }

            del model
            gc.collect()

        fold_summary.append(fold_info)

        del X_tr, y_tr, X_va, y_va
        gc.collect()

    save_json(MODEL_DIR / "oof_fold_summary.json", {"folds": fold_summary})

    # -----------------------------------------------------
    # Step 2: Train meta-learner on OOF predictions
    # -----------------------------------------------------
    print("\n[STEP 2] Training meta-learner on OOF predictions...")
    meta_model = build_model(META_MODEL_NAME)
    meta_model.fit(oof_meta_X, y_train)

    # -----------------------------------------------------
    # Step 3: Fit final base models on full training split
    # -----------------------------------------------------
    print("\n[STEP 3] Training final base models on full training split...")
    final_base_models = {}
    val_meta_X = np.zeros((len(X_val), len(BASE_MODEL_ORDER)), dtype=np.float32)
    test_meta_X = np.zeros((len(X_test), len(BASE_MODEL_ORDER)), dtype=np.float32)

    base_model_metrics = {}

    for model_idx, model_name in enumerate(BASE_MODEL_ORDER):
        print(f"[FINAL] Training {model_name} on full training split...")
        model = build_model(model_name)
        model = fit_model(model_name, model, X_train, y_train)

        val_attack_probs = get_attack_proba(model, X_val, attack_class_id)
        test_attack_probs = get_attack_proba(model, X_test, attack_class_id)

        val_meta_X[:, model_idx] = val_attack_probs
        test_meta_X[:, model_idx] = test_attack_probs

        base_threshold_info = choose_best_threshold(y_val, val_attack_probs, attack_class_id)

        base_val_metrics = evaluate_with_probs(
            split_name="val",
            y_true=y_val,
            attack_probs=val_attack_probs,
            attack_threshold=base_threshold_info["best_threshold"],
            attack_class_id=attack_class_id,
            benign_class_id=benign_class_id,
        )

        base_test_metrics = evaluate_with_probs(
            split_name="test",
            y_true=y_test,
            attack_probs=test_attack_probs,
            attack_threshold=base_threshold_info["best_threshold"],
            attack_class_id=attack_class_id,
            benign_class_id=benign_class_id,
        )

        base_model_metrics[model_name] = {
            "threshold_info": base_threshold_info,
            "val_metrics": base_val_metrics,
            "test_metrics": base_test_metrics,
        }

        final_base_models[model_name] = model

    save_json(MODEL_DIR / "base_model_metrics.json", base_model_metrics)

    # -----------------------------------------------------
    # Step 4: Stack on validation/test
    # -----------------------------------------------------
    print("\n[STEP 4] Generating stacked probabilities...")
    val_stack_attack_probs = get_attack_proba(meta_model, val_meta_X, attack_class_id)
    test_stack_attack_probs = get_attack_proba(meta_model, test_meta_X, attack_class_id)

    stack_threshold_info = choose_best_threshold(y_val, val_stack_attack_probs, attack_class_id)

    stack_val_metrics = evaluate_with_probs(
        split_name="val",
        y_true=y_val,
        attack_probs=val_stack_attack_probs,
        attack_threshold=stack_threshold_info["best_threshold"],
        attack_class_id=attack_class_id,
        benign_class_id=benign_class_id,
    )

    stack_test_metrics = evaluate_with_probs(
        split_name="test",
        y_true=y_test,
        attack_probs=test_stack_attack_probs,
        attack_threshold=stack_threshold_info["best_threshold"],
        attack_class_id=attack_class_id,
        benign_class_id=benign_class_id,
    )

    # -----------------------------------------------------
    # Step 5: Save all models and artifacts
    # -----------------------------------------------------
    print("\n[STEP 5] Saving models and artifacts...")
    for model_name, model in final_base_models.items():
        joblib.dump(model, MODEL_DIR / f"{model_name}_base_model.joblib")

    joblib.dump(meta_model, MODEL_DIR / "meta_logreg_model.joblib")

    stacking_config = {
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "feature_columns": feature_cols,
        "meta_feature_names": meta_feature_names,
        "label_mapping": label_map,
        "attack_class_id": attack_class_id,
        "benign_class_id": benign_class_id,
        "base_model_order": BASE_MODEL_ORDER,
        "rf_params": RF_PARAMS,
        "hgb_params": HGB_PARAMS,
        "lr_params": LR_PARAMS,
        "xgb_params": XGB_PARAMS,
        "meta_lr_params": META_LR_PARAMS,
    }

    inference_schema = {
        "raw_feature_columns": feature_cols,
        "meta_feature_names": meta_feature_names,
        "label_mapping": label_map,
        "attack_class_id": attack_class_id,
        "benign_class_id": benign_class_id,
        "base_model_order": BASE_MODEL_ORDER,
        "base_model_outputs": "Each base model contributes one feature: probability of ATTACK.",
        "final_threshold": stack_threshold_info["best_threshold"],
    }

    save_json(MODEL_DIR / "stacking_config.json", stacking_config)
    save_json(MODEL_DIR / "stack_threshold_info.json", stack_threshold_info)
    save_json(MODEL_DIR / "stack_val_metrics.json", stack_val_metrics)
    save_json(MODEL_DIR / "stack_test_metrics.json", stack_test_metrics)
    save_json(MODEL_DIR / "inference_schema.json", inference_schema)

    # Optional: save meta feature matrices for inspection/debugging
    pd.DataFrame(oof_meta_X, columns=meta_feature_names).assign(target_id=y_train).to_csv(
        MODEL_DIR / "oof_meta_train.csv", index=False
    )
    pd.DataFrame(val_meta_X, columns=meta_feature_names).assign(target_id=y_val).to_csv(
        MODEL_DIR / "meta_val.csv", index=False
    )
    pd.DataFrame(test_meta_X, columns=meta_feature_names).assign(target_id=y_test).to_csv(
        MODEL_DIR / "meta_test.csv", index=False
    )

    print("\n[DONE] Full stacking ensemble training complete.")
    print(f"[SAVE] {MODEL_DIR / 'stack_val_metrics.json'}")
    print(f"[SAVE] {MODEL_DIR / 'stack_test_metrics.json'}")
    print(f"[SAVE] {MODEL_DIR / 'base_model_metrics.json'}")
    print(f"[SAVE] {MODEL_DIR / 'stack_threshold_info.json'}")
    print(f"[SAVE] {MODEL_DIR / 'inference_schema.json'}")

    print("\n[STACK SUMMARY]")
    print("Validation F1 (ATTACK):", round(stack_val_metrics["f1_attack"], 6))
    print("Validation Recall (ATTACK):", round(stack_val_metrics["recall_attack"], 6))
    print("Validation Precision (ATTACK):", round(stack_val_metrics["precision_attack"], 6))
    print("Validation ROC-AUC (ATTACK):", round(stack_val_metrics["roc_auc_attack"], 6))
    print("Validation AP (ATTACK):", round(stack_val_metrics["average_precision_attack"], 6))

    print("Test F1 (ATTACK):", round(stack_test_metrics["f1_attack"], 6))
    print("Test Recall (ATTACK):", round(stack_test_metrics["recall_attack"], 6))
    print("Test Precision (ATTACK):", round(stack_test_metrics["precision_attack"], 6))
    print("Test ROC-AUC (ATTACK):", round(stack_test_metrics["roc_auc_attack"], 6))
    print("Test AP (ATTACK):", round(stack_test_metrics["average_precision_attack"], 6))


if __name__ == "__main__":
    main()
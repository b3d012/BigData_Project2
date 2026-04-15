from __future__ import annotations

import json
from pathlib import Path


BASELINE_DIR = Path("models/baseline_xgb")
STACK_DIR = Path("models/stacking_ensemble")
OUT_PATH = Path("outputs/confusion_matrices_summary.json")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def matrix_breakdown(matrix: list[list[int]]) -> dict:
    # Matrix format is labels=[attack, benign]
    # [[TP, FN],
    #  [FP, TN]]
    tp = int(matrix[0][0])
    fn = int(matrix[0][1])
    fp = int(matrix[1][0])
    tn = int(matrix[1][1])

    return {
        "matrix_labels_order": "[attack, benign]",
        "matrix": matrix,
        "tp_attack": tp,
        "fn_attack": fn,
        "fp_attack": fp,
        "tn_attack": tn,
    }


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    xgb_val = load_json(BASELINE_DIR / "val_metrics.json")
    xgb_test = load_json(BASELINE_DIR / "test_metrics.json")

    stack_val = load_json(STACK_DIR / "stack_val_metrics.json")
    stack_test = load_json(STACK_DIR / "stack_test_metrics.json")

    base_metrics = load_json(STACK_DIR / "base_model_metrics.json")

    summary = {
        "baseline_xgboost": {
            "val": matrix_breakdown(xgb_val["confusion_matrix_labels_[attack,benign]"]),
            "test": matrix_breakdown(xgb_test["confusion_matrix_labels_[attack,benign]"]),
        },
        "stacked_ensemble_final": {
            "val": matrix_breakdown(stack_val["confusion_matrix_labels_[attack,benign]"]),
            "test": matrix_breakdown(stack_test["confusion_matrix_labels_[attack,benign]"]),
        },
        "base_models_inside_stack": {},
    }

    for model_name, model_info in base_metrics.items():
        summary["base_models_inside_stack"][model_name] = {
            "val": matrix_breakdown(model_info["val_metrics"]["confusion_matrix_labels_[attack,benign]"]),
            "test": matrix_breakdown(model_info["test_metrics"]["confusion_matrix_labels_[attack,benign]"]),
        }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[DONE] Confusion matrix summary exported.")
    print(f"[SAVE] {OUT_PATH}")

    print("\n=== Baseline XGBoost (test) ===")
    print(summary["baseline_xgboost"]["test"])

    print("\n=== Stacked Ensemble Final (test) ===")
    print(summary["stacked_ensemble_final"]["test"])

    print("\n=== Base Models Inside Stack (test) ===")
    for model_name, info in summary["base_models_inside_stack"].items():
        print(f"{model_name}: {info['test']}")


if __name__ == "__main__":
    main()
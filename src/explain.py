"""Generate SHAP global and local explanations for CV-selected models."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt

from src.preprocessing import FEATURE_SCHEMAS, load_dataset, transformed_feature_mapping
from src.train import DEFAULT_SEED


matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
BACKGROUND_SIZE = 50
RANDOM_SEED = 42


def select_model_from_cv(training_report: dict[str, Any]) -> tuple[str, dict[str, float]]:
    """Select the highest mean training-CV ROC-AUC; ties use lower fold std."""
    model_scores: dict[str, dict[str, float]] = {}
    for model_name, model_report in training_report["models"].items():
        roc_auc = model_report["cross_validation"]["metrics"]["roc_auc"]
        model_scores[model_name] = {
            "mean": float(roc_auc["mean"]),
            "std": float(roc_auc["std"]),
        }
    selected = max(
        model_scores,
        key=lambda name: (model_scores[name]["mean"], -model_scores[name]["std"]),
    )
    return selected, model_scores[selected]


def normalize_shap_outputs(
    raw_values: Any,
    raw_base_values: Any,
    classes: np.ndarray,
    row_count: int,
    feature_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize SHAP list/array layouts to positive-class values and base values."""
    class_labels = np.asarray(classes)
    positive_indices = np.flatnonzero(class_labels == 1)
    if len(positive_indices) != 1:
        raise ValueError(f"Model must have one positive class labeled 1: {class_labels}.")
    positive_index = int(positive_indices[0])

    if isinstance(raw_values, list):
        if len(raw_values) == len(class_labels):
            raw_values = raw_values[positive_index]
        elif len(raw_values) == 1:
            raw_values = raw_values[0]
        else:
            raise ValueError(
                f"Unexpected SHAP output list length {len(raw_values)} for "
                f"classes {class_labels.tolist()}."
            )
    values = np.asarray(raw_values, dtype=float)
    if values.ndim == 3:
        if values.shape == (row_count, feature_count, len(class_labels)):
            values = values[:, :, positive_index]
        elif values.shape == (len(class_labels), row_count, feature_count):
            values = values[positive_index, :, :]
        elif values.shape == (row_count, len(class_labels), feature_count):
            values = values[:, positive_index, :]
    if values.shape != (row_count, feature_count):
        raise ValueError(
            f"Unexpected normalized SHAP value shape {values.shape}; expected "
            f"{(row_count, feature_count)}."
        )

    if isinstance(raw_base_values, list) and len(raw_base_values) == len(class_labels):
        raw_base_values = raw_base_values[positive_index]
    base_values = np.asarray(raw_base_values, dtype=float)
    if base_values.ndim == 0:
        base_values = np.full(row_count, float(base_values))
    elif base_values.ndim == 1:
        if len(base_values) == len(class_labels):
            base_values = np.full(row_count, float(base_values[positive_index]))
        elif len(base_values) == 1:
            base_values = np.full(row_count, float(base_values[0]))
        elif len(base_values) != row_count:
            raise ValueError(f"Unexpected SHAP base-value shape {base_values.shape}.")
    elif base_values.ndim == 2:
        if base_values.shape == (row_count, len(class_labels)):
            base_values = base_values[:, positive_index]
        elif base_values.shape == (len(class_labels), row_count):
            base_values = base_values[positive_index, :]
        else:
            raise ValueError(f"Unexpected SHAP base-value shape {base_values.shape}.")
    else:
        raise ValueError(f"Unexpected SHAP base-value shape {base_values.shape}.")

    if not np.isfinite(values).all() or not np.isfinite(base_values).all():
        raise ValueError("SHAP values and base values must be finite.")
    return values, np.asarray(base_values, dtype=float)


def _create_explainer(
    model_name: str,
    estimator: Any,
    background: np.ndarray,
) -> tuple[Any, str]:
    if model_name == "logistic_regression":
        masker = shap.maskers.Independent(background)
        return shap.LinearExplainer(estimator, masker), "raw decision-function log-odds"
    if model_name == "xgboost":
        explainer = shap.TreeExplainer(
            estimator,
            data=background,
            feature_perturbation="interventional",
            model_output="raw",
        )
        return explainer, "raw tree margin (log-odds)"
    if model_name == "random_forest":
        explainer = shap.TreeExplainer(
            estimator,
            data=background,
            feature_perturbation="interventional",
            model_output="raw",
        )
        return explainer, "raw class probability"
    raise ValueError(f"Unsupported SHAP model type: {model_name!r}")


def _model_output_in_explanation_space(
    model_name: str,
    estimator: Any,
    transformed_row: np.ndarray,
) -> float:
    if model_name == "logistic_regression":
        return float(np.asarray(estimator.decision_function(transformed_row)).reshape(-1)[0])
    if model_name == "xgboost":
        return float(np.asarray(estimator.predict(transformed_row, output_margin=True)).reshape(-1)[0])
    classes = np.asarray(estimator.classes_)
    positive_index = int(np.flatnonzero(classes == 1)[0])
    return float(estimator.predict_proba(transformed_row)[0, positive_index])


def _save_global_plots(
    shap_values: np.ndarray,
    transformed_test: np.ndarray,
    transformed_names: list[str],
    source_feature_mapping: dict[str, str],
    output_dir: Path,
    disease: str,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    beeswarm_path = output_dir / "shap_summary_beeswarm.png"
    shap.summary_plot(
        shap_values,
        transformed_test,
        feature_names=transformed_names,
        max_display=min(20, len(transformed_names)),
        show=False,
    )
    plt.gcf().set_size_inches(10, 7)
    plt.gcf().savefig(beeswarm_path, dpi=160, bbox_inches="tight")
    plt.close(plt.gcf())

    grouped_values: dict[str, list[int]] = {}
    for index, transformed_name in enumerate(transformed_names):
        source_name = source_feature_mapping.get(transformed_name, transformed_name)
        grouped_values.setdefault(source_name, []).append(index)
    global_importance: list[dict[str, Any]] = []
    for source_name, indices in grouped_values.items():
        per_row_magnitude = np.abs(shap_values[:, indices]).sum(axis=1)
        global_importance.append(
            {
                "feature": source_name,
                "mean_absolute_shap": float(per_row_magnitude.mean()),
            }
        )
    global_importance.sort(key=lambda item: item["mean_absolute_shap"], reverse=True)

    importance_path = output_dir / "shap_global_importance.png"
    top_features = global_importance[: min(20, len(global_importance))]
    fig, axis = plt.subplots(figsize=(9, max(4, 0.35 * len(top_features))), constrained_layout=True)
    axis.barh(
        [item["feature"] for item in reversed(top_features)],
        [item["mean_absolute_shap"] for item in reversed(top_features)],
        color="#2878a5",
    )
    axis.set(
        title=f"{disease.title()} mean absolute SHAP by original feature",
        xlabel="Mean absolute contribution in model output space",
        ylabel="Original feature",
    )
    fig.savefig(importance_path, dpi=160)
    plt.close(fig)
    return (
        {"summary_beeswarm": str(beeswarm_path), "global_importance": str(importance_path)},
        global_importance,
    )


def explain_disease(
    disease: str,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Explain the CV-selected individual model using training background data."""
    if disease not in FEATURE_SCHEMAS:
        raise ValueError(f"Unknown disease: {disease!r}")

    disease_results_dir = results_dir / disease
    training_path = disease_results_dir / "training.json"
    split_path = disease_results_dir / "data_split.json"
    if not training_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(f"Missing training/split report for {disease}.")
    training_report = json.loads(training_path.read_text(encoding="utf-8"))
    split_report = json.loads(split_path.read_text(encoding="utf-8"))
    selected_model, selected_cv = select_model_from_cv(training_report)
    model_path = model_dir / disease / f"{selected_model}.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing selected fitted model: {model_path}")

    output_dir = disease_results_dir / "shap"
    report_path = output_dir / "explanation_report.json"
    contributions_path = output_dir / "local_contributions.csv"
    expected_outputs = (
        report_path,
        contributions_path,
        output_dir / "shap_summary_beeswarm.png",
        output_dir / "shap_global_importance.png",
        output_dir / "shap_local_waterfall.png",
    )
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "SHAP artifacts already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    dataset = load_dataset(dataset_dir / f"{disease}.csv")
    train_indices = np.asarray(split_report["training_indices"], dtype=int)
    test_indices = np.asarray(split_report["test_indices"], dtype=int)
    x_train = dataset.features.iloc[train_indices]
    x_test = dataset.features.iloc[test_indices]
    model = joblib.load(model_path)
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    classes = np.asarray(estimator.classes_)
    if len(np.flatnonzero(classes == 1)) != 1:
        raise ValueError(f"Selected model does not expose a positive class labeled 1.")

    background_rows = min(BACKGROUND_SIZE, len(x_train))
    background_frame = x_train.sample(n=background_rows, random_state=RANDOM_SEED)
    background = np.asarray(preprocessor.transform(background_frame), dtype=float)
    transformed_test = np.asarray(preprocessor.transform(x_test), dtype=float)
    transformed_names = preprocessor.get_feature_names_out().tolist()
    if transformed_test.shape[1] != len(transformed_names):
        raise ValueError("Transformed data and feature names have different dimensions.")
    source_feature_mapping = transformed_feature_mapping(preprocessor)
    explainer, output_space = _create_explainer(selected_model, estimator, background)
    raw_shap_values = explainer.shap_values(transformed_test)
    shap_values, base_values = normalize_shap_outputs(
        raw_shap_values,
        explainer.expected_value,
        classes,
        row_count=len(x_test),
        feature_count=transformed_test.shape[1],
    )

    local_row = 0
    local_values = shap_values[local_row]
    local_base_value = float(base_values[local_row])
    local_output = _model_output_in_explanation_space(
        selected_model, estimator, transformed_test[local_row : local_row + 1]
    )
    reconstructed_output = local_base_value + float(local_values.sum())
    additivity_error = abs(local_output - reconstructed_output)

    explanation = shap.Explanation(
        values=local_values,
        base_values=local_base_value,
        data=transformed_test[local_row],
        feature_names=transformed_names,
    )
    waterfall_path = output_dir / "shap_local_waterfall.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    shap.plots.waterfall(
        explanation,
        max_display=min(15, len(transformed_names)),
        show=False,
    )
    plt.gcf().set_size_inches(10, 7)
    plt.gcf().savefig(waterfall_path, dpi=160, bbox_inches="tight")
    plt.close(plt.gcf())

    plot_paths, global_importance = _save_global_plots(
        shap_values,
        transformed_test,
        transformed_names,
        source_feature_mapping,
        output_dir,
        disease,
    )
    plot_paths["local_waterfall"] = str(waterfall_path)

    local_contributions = pd.DataFrame(
        {
            "transformed_feature": transformed_names,
            "source_feature": [
                source_feature_mapping.get(name, name) for name in transformed_names
            ],
            "original_feature_name": [
                dataset.feature_name_mapping.get(
                    source_feature_mapping.get(name, name),
                    source_feature_mapping.get(name, name),
                )
                for name in transformed_names
            ],
            "transformed_value": transformed_test[local_row],
            "shap_value": local_values,
        }
    )
    local_contributions["direction"] = np.where(
        local_contributions["shap_value"] >= 0,
        "increases model output",
        "decreases model output",
    )
    local_contributions["absolute_shap"] = local_contributions["shap_value"].abs()
    local_contributions = local_contributions.sort_values(
        "absolute_shap", ascending=False
    )
    local_contributions.to_csv(contributions_path, index=False)

    report = {
        "disease": disease,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "shap_version": shap.__version__,
        "selected_model": selected_model,
        "selection_criterion": "highest mean 5-fold training CV ROC-AUC; ties use lower standard deviation",
        "selection_cv_roc_auc": selected_cv,
        "model_output_space": output_space,
        "positive_class_label": 1,
        "explanation_population": "reserved test partition; explanations do not tune the model",
        "explained_test_rows": int(len(x_test)),
        "background_rows_from_training_partition": int(background_rows),
        "transformed_feature_count": int(len(transformed_names)),
        "transformed_to_source_feature_mapping": source_feature_mapping,
        "local_example": {
            "cleaned_dataset_row_index": int(test_indices[local_row]),
            "base_value": local_base_value,
            "model_output": local_output,
            "base_plus_shap_sum": reconstructed_output,
            "absolute_additivity_error": additivity_error,
            "raw_feature_values": {
                str(column): None if pd.isna(value) else str(value)
                for column, value in x_test.iloc[local_row].items()
            },
            "top_contributions": local_contributions.head(15).to_dict(
                orient="records"
            ),
        },
        "global_importance": global_importance,
        "plots": plot_paths,
        "interpretation_limitations": (
            "SHAP values describe this fitted model's behavior in the stated output "
            "space. They do not establish medical causation, clinical correctness, "
            "fairness, or clinical validity."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease", choices=tuple(FEATURE_SCHEMAS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    diseases = [args.disease] if args.disease else list(FEATURE_SCHEMAS)
    try:
        for disease in diseases:
            report = explain_disease(disease, overwrite=args.overwrite)
            print(
                f"{disease}: explained {report['selected_model']} in "
                f"{report['model_output_space']} space; "
                f"local additivity error={report['local_example']['absolute_additivity_error']:.3g}"
            )
    except (FileExistsError, FileNotFoundError, TypeError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
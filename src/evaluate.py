"""Evaluate saved disease models on their reserved test partitions."""

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
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.compare_models import BASE_MODEL_NAMES, WeightedSoftVotingClassifier
from src.preprocessing import FEATURE_SCHEMAS, load_dataset
from src.train import DEFAULT_CV_FOLDS, DEFAULT_SEED


matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
ENSEMBLE_NAME = "weighted_soft_voting"
EVALUATED_MODELS = (*BASE_MODEL_NAMES, ENSEMBLE_NAME)


def select_youden_threshold(
    target: np.ndarray | pd.Series,
    positive_probability: np.ndarray,
) -> tuple[float, float]:
    """Select the threshold maximizing sensitivity + specificity - 1."""
    labels = np.asarray(target, dtype=int)
    probabilities = np.asarray(positive_probability, dtype=float)
    if labels.ndim != 1 or probabilities.shape != labels.shape or not len(labels):
        raise ValueError("Target and probability inputs must be non-empty 1D arrays.")
    if not np.isin(labels, [0, 1]).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Threshold selection requires both binary classes 0 and 1.")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise ValueError("Positive-class probabilities must be finite values in [0, 1].")

    false_positive_rate, sensitivity, thresholds = roc_curve(labels, probabilities)
    youden_j = sensitivity - false_positive_rate
    finite_candidates = np.isfinite(thresholds)
    if not finite_candidates.any():
        raise ValueError("ROC threshold search returned no finite thresholds.")
    best_j = float(np.max(youden_j[finite_candidates]))
    best_mask = finite_candidates & np.isclose(youden_j, best_j)
    candidates = thresholds[best_mask]
    threshold = float(candidates[np.argmin(np.abs(candidates - 0.5))])
    return threshold, best_j


def compute_classification_metrics(
    target: np.ndarray | pd.Series,
    positive_probability: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Calculate thresholded and probability-based binary classification metrics."""
    labels = np.asarray(target, dtype=int)
    probabilities = np.asarray(positive_probability, dtype=float)
    if labels.ndim != 1 or probabilities.shape != labels.shape or not len(labels):
        raise ValueError("Target and probability inputs must be non-empty 1D arrays.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Classification threshold must be finite and in [0, 1].")
    if not np.isin(labels, [0, 1]).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Evaluation requires both binary classes 0 and 1.")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise ValueError("Positive-class probabilities must be finite values in [0, 1].")

    predictions = (probabilities >= threshold).astype("int64")
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    sensitivity = float(recall_score(labels, predictions, zero_division=0))
    specificity_denominator = int(true_negative + false_positive)
    specificity = (
        float(true_negative / specificity_denominator)
        if specificity_denominator
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "false_negative_rate": float(1.0 - sensitivity),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix": [
            [int(true_negative), int(false_positive)],
            [int(false_negative), int(true_positive)],
        ],
        "counts": {
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
        },
    }


def _positive_probability(estimator: Any, features: pd.DataFrame) -> np.ndarray:
    classes = np.asarray(estimator.classes_)
    positive_indices = np.flatnonzero(classes == 1)
    if len(positive_indices) != 1:
        raise ValueError(f"Model must expose one positive class labeled 1; got {classes}.")
    probabilities = np.asarray(estimator.predict_proba(features), dtype=float)
    if probabilities.shape != (len(features), len(classes)):
        raise ValueError(f"Invalid model probability output shape: {probabilities.shape}.")
    return probabilities[:, positive_indices[0]]


def _load_partitions(
    disease: str,
    dataset_dir: Path,
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict[str, Any]]:
    dataset = load_dataset(dataset_dir / f"{disease}.csv")
    split_path = results_dir / disease / "data_split.json"
    training_path = results_dir / disease / "training.json"
    if not split_path.is_file() or not training_path.is_file():
        raise FileNotFoundError(f"Missing saved training split for {disease}.")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    training_report = json.loads(training_path.read_text(encoding="utf-8"))
    train_indices = np.asarray(split["training_indices"], dtype=int)
    test_indices = np.asarray(split["test_indices"], dtype=int)
    all_indices = set(range(len(dataset.target)))
    if (
        len(np.unique(train_indices)) != len(train_indices)
        or len(np.unique(test_indices)) != len(test_indices)
        or set(train_indices) & set(test_indices)
        or set(train_indices) | set(test_indices) != all_indices
    ):
        raise ValueError(f"Saved row split for {disease} is invalid or incomplete.")
    if len(train_indices) != training_report["training_samples"]:
        raise ValueError(f"Saved training row count for {disease} is inconsistent.")
    if len(test_indices) != training_report["test_samples_reserved"]:
        raise ValueError(f"Saved test row count for {disease} is inconsistent.")

    x_train = dataset.features.iloc[train_indices]
    y_train = dataset.target.iloc[train_indices]
    x_test = dataset.features.iloc[test_indices]
    y_test = dataset.target.iloc[test_indices]
    return x_train, y_train, x_test, y_test, training_report


def _save_curves(
    disease: str,
    output_dir: Path,
    target: np.ndarray,
    test_probabilities: dict[str, np.ndarray],
    evaluation: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    roc_path = output_dir / "test_roc_curves.png"
    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for model_name, probabilities in test_probabilities.items():
        false_positive_rate, sensitivity, _ = roc_curve(target, probabilities)
        axis.plot(
            false_positive_rate,
            sensitivity,
            label=f"{model_name} (AUC {roc_auc_score(target, probabilities):.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.set(title=f"{disease.title()} test ROC curves", xlabel="False-positive rate", ylabel="Sensitivity")
    axis.legend(loc="lower right", fontsize="small")
    fig.savefig(roc_path, dpi=160)
    plt.close(fig)

    pr_path = output_dir / "test_precision_recall_curves.png"
    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for model_name, probabilities in test_probabilities.items():
        precision, recall, _ = precision_recall_curve(target, probabilities)
        axis.plot(
            recall,
            precision,
            label=f"{model_name} (AP {average_precision_score(target, probabilities):.3f})",
        )
    axis.set(title=f"{disease.title()} test precision-recall curves", xlabel="Recall / sensitivity", ylabel="Precision")
    axis.legend(loc="lower left", fontsize="small")
    fig.savefig(pr_path, dpi=160)
    plt.close(fig)

    calibration_path = output_dir / "test_calibration_curves.png"
    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal calibration")
    for model_name, probabilities in test_probabilities.items():
        observed_fraction, mean_probability = calibration_curve(
            target,
            probabilities,
            n_bins=min(10, max(2, len(target) // 5)),
            strategy="quantile",
        )
        axis.plot(mean_probability, observed_fraction, marker="o", label=model_name)
    axis.set(title=f"{disease.title()} test calibration", xlabel="Mean predicted positive-class score", ylabel="Observed positive fraction")
    axis.legend(loc="upper left", fontsize="small")
    fig.savefig(calibration_path, dpi=160)
    plt.close(fig)

    confusion_path = output_dir / "test_confusion_matrices.png"
    fig, axes = plt.subplots(
        nrows=len(EVALUATED_MODELS), ncols=2, figsize=(9, 3.2 * len(EVALUATED_MODELS))
    )
    for row_index, model_name in enumerate(EVALUATED_MODELS):
        for column_index, scenario in enumerate(("default_0_5", "youden_threshold")):
            metrics = evaluation["models"][model_name]["test_metrics"][scenario]
            axis = axes[row_index, column_index]
            sns.heatmap(
                np.asarray(metrics["confusion_matrix"], dtype=int),
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["Predicted 0", "Predicted 1"],
                yticklabels=["Actual 0", "Actual 1"],
                ax=axis,
            )
            axis.set_title(f"{model_name}: {scenario.replace('_', ' ')}")
            axis.set_xlabel("")
            axis.set_ylabel("")
    fig.suptitle(f"{disease.title()} test confusion matrices", y=1.002)
    fig.tight_layout()
    fig.savefig(confusion_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "roc_curves": str(roc_path),
        "precision_recall_curves": str(pr_path),
        "calibration_curves": str(calibration_path),
        "confusion_matrices": str(confusion_path),
    }


def evaluate_disease(
    disease: str,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Select thresholds on training OOF scores, then evaluate the fixed test set."""
    if disease not in FEATURE_SCHEMAS:
        raise ValueError(f"Unknown disease: {disease!r}")

    x_train, y_train, x_test, y_test, training_report = _load_partitions(
        disease, dataset_dir, results_dir
    )
    disease_model_dir = model_dir / disease
    disease_results_dir = results_dir / disease
    evaluation_path = disease_results_dir / "evaluation.json"
    metrics_path = disease_results_dir / "test_metrics.csv"
    figures_dir = disease_results_dir / "figures"
    expected_figures = [
        figures_dir / "test_roc_curves.png",
        figures_dir / "test_precision_recall_curves.png",
        figures_dir / "test_calibration_curves.png",
        figures_dir / "test_confusion_matrices.png",
    ]
    existing = [
        path for path in (evaluation_path, metrics_path, *expected_figures) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "Evaluation artifacts already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    models: dict[str, Any] = {}
    for model_name in BASE_MODEL_NAMES:
        model_path = disease_model_dir / f"{model_name}.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing fitted model: {model_path}")
        models[model_name] = joblib.load(model_path)
    ensemble_path = disease_model_dir / "weighted_ensemble.joblib"
    if not ensemble_path.is_file():
        raise FileNotFoundError(f"Missing weighted ensemble: {ensemble_path}")
    models[ENSEMBLE_NAME] = joblib.load(ensemble_path)
    if not isinstance(models[ENSEMBLE_NAME], WeightedSoftVotingClassifier):
        raise TypeError("Saved weighted ensemble has an unexpected model type.")

    seed = int(training_report["seed"])
    cv_folds = int(training_report["cross_validation"]["folds"])
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    training_probabilities: dict[str, np.ndarray] = {}
    for model_name in BASE_MODEL_NAMES:
        oof_matrix = cross_val_predict(
            models[model_name],
            x_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )
        classes = np.asarray(models[model_name].classes_)
        positive_indices = np.flatnonzero(classes == 1)
        if len(positive_indices) != 1:
            raise ValueError(f"{model_name} has no unique positive class labeled 1.")
        training_probabilities[model_name] = oof_matrix[:, positive_indices[0]]
    training_probabilities[ENSEMBLE_NAME] = sum(
        models[ENSEMBLE_NAME].weights[model_name]
        * training_probabilities[model_name]
        for model_name in BASE_MODEL_NAMES
    )

    threshold_selection: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    for model_name, probabilities in training_probabilities.items():
        threshold, youden_j = select_youden_threshold(y_train, probabilities)
        thresholds[model_name] = threshold
        threshold_selection[model_name] = {
            "threshold": threshold,
            "method": "maximize Youden J = sensitivity + specificity - 1",
            "training_validation_predictions": "StratifiedKFold out-of-fold predictions",
            "training_oof_youden_j": youden_j,
        }

    test_probabilities = {
        model_name: _positive_probability(model, x_test)
        for model_name, model in models.items()
    }
    evaluation: dict[str, Any] = {
        "disease": disease,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_partition": {
            "rows": int(len(y_test)),
            "selection_used_test_data": False,
            "threshold_selection_partition": "training split only, using stratified OOF predictions",
            "split_seed": seed,
        },
        "threshold_selection": {
            "method": "Youden J on training-only out-of-fold probabilities; ties choose the threshold closest to 0.5",
            "note": (
                "Thresholds were fixed before test predictions were evaluated. "
                "For the weighted ensemble, OOF weights and threshold were both "
                "derived from training data; the held-out set remains the final "
                "generalization check."
            ),
            "models": threshold_selection,
        },
        "models": {},
        "clinical_interpretation": (
            "Predictions and scores are research outputs, not clinically validated "
            "risk estimates or diagnoses."
        ),
    }
    csv_rows: list[dict[str, Any]] = []
    for model_name in EVALUATED_MODELS:
        default_metrics = compute_classification_metrics(
            y_test, test_probabilities[model_name], 0.5
        )
        selected_metrics = compute_classification_metrics(
            y_test, test_probabilities[model_name], thresholds[model_name]
        )
        evaluation["models"][model_name] = {
            "threshold_selection": threshold_selection[model_name],
            "test_metrics": {
                "default_0_5": default_metrics,
                "youden_threshold": selected_metrics,
            },
        }
        for scenario_name, metrics in (
            ("default_0_5", default_metrics),
            ("youden_threshold", selected_metrics),
        ):
            csv_rows.append(
                {
                    "disease": disease,
                    "model": model_name,
                    "threshold_scenario": scenario_name,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key not in {"confusion_matrix", "counts"}
                    },
                    **metrics["counts"],
                }
            )

    evaluation["plots"] = _save_curves(
        disease,
        figures_dir,
        y_test.to_numpy(),
        test_probabilities,
        evaluation,
    )
    disease_results_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = list(csv_rows[0])
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    return evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease", choices=tuple(FEATURE_SCHEMAS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    diseases = [args.disease] if args.disease else list(FEATURE_SCHEMAS)
    try:
        for disease in diseases:
            report = evaluate_disease(disease, overwrite=args.overwrite)
            print(f"{disease}: final held-out evaluation complete")
            for model_name, model_report in report["models"].items():
                metrics = model_report["test_metrics"]["default_0_5"]
                print(
                    f"  {model_name}: accuracy={metrics['accuracy']:.3f}, "
                    f"sensitivity={metrics['sensitivity']:.3f}, "
                    f"specificity={metrics['specificity']:.3f}, "
                    f"ROC-AUC={metrics['roc_auc']:.3f}, "
                    f"Brier={metrics['brier_score']:.3f}"
                )
    except (FileExistsError, FileNotFoundError, TypeError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
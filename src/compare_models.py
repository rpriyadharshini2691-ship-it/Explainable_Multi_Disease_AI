"""Derive training-only weighted ensembles and comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.preprocessing import FEATURE_SCHEMAS, load_dataset
from src.train import DEFAULT_CV_FOLDS, DEFAULT_SEED


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
BASE_MODEL_NAMES = ("logistic_regression", "random_forest", "xgboost")
CV_METRICS = (
    "accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "f1",
    "roc_auc",
    "average_precision",
    "brier_score",
)


class WeightedSoftVotingClassifier:
    """Average fitted base-model probabilities with validated class alignment."""

    def __init__(
        self,
        estimators: dict[str, Any],
        weights: dict[str, float],
    ) -> None:
        if set(estimators) != set(weights):
            raise ValueError("Estimator names and ensemble weight names must match.")
        if not estimators:
            raise ValueError("At least one fitted estimator is required.")
        weight_values = np.asarray([weights[name] for name in estimators], dtype=float)
        if not np.isfinite(weight_values).all() or (weight_values < 0).any():
            raise ValueError("Ensemble weights must be finite and non-negative.")
        if not np.isclose(weight_values.sum(), 1.0, atol=1e-8):
            raise ValueError("Ensemble weights must sum to 1.")

        self.estimators = estimators
        self.weights = {name: float(weights[name]) for name in estimators}
        self.classes_ = np.asarray([0, 1])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        positive_probability = np.zeros(len(features), dtype=float)
        for model_name, estimator in self.estimators.items():
            classes = np.asarray(estimator.classes_)
            positive_indices = np.flatnonzero(classes == 1)
            negative_indices = np.flatnonzero(classes == 0)
            if len(positive_indices) != 1 or len(negative_indices) != 1:
                raise ValueError(
                    f"{model_name} must expose exactly classes 0 and 1; "
                    f"got {classes.tolist()}."
                )
            probabilities = np.asarray(estimator.predict_proba(features), dtype=float)
            if probabilities.shape != (len(features), len(classes)):
                raise ValueError(
                    f"{model_name} returned an invalid probability shape: "
                    f"{probabilities.shape}."
                )
            positive_probability += (
                self.weights[model_name] * probabilities[:, positive_indices[0]]
            )

        positive_probability = np.clip(positive_probability, 0.0, 1.0)
        return np.column_stack((1.0 - positive_probability, positive_probability))

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(features)[:, 1] >= 0.5).astype("int64")


def derive_ensemble_weights(
    out_of_fold_probabilities: np.ndarray,
    target: Sequence[int],
    model_names: Sequence[str] = BASE_MODEL_NAMES,
) -> dict[str, float]:
    """Minimize OOF Brier loss subject to nonnegative weights summing to one."""
    probabilities = np.asarray(out_of_fold_probabilities, dtype=float)
    labels = np.asarray(target, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape != (len(labels), len(model_names)):
        raise ValueError(
            "OOF probabilities must have one row per label and one column per model."
        )
    if not len(labels) or not np.isfinite(probabilities).all():
        raise ValueError("OOF probabilities and labels must be non-empty and finite.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("OOF probabilities must lie in [0, 1].")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Ensemble weight derivation requires binary 0/1 labels.")

    model_count = len(model_names)
    initial_weights = np.full(model_count, 1.0 / model_count)

    def objective(weights: np.ndarray) -> float:
        residual = probabilities @ weights - labels
        return float(np.mean(residual**2))

    def gradient(weights: np.ndarray) -> np.ndarray:
        residual = probabilities @ weights - labels
        return 2.0 * probabilities.T @ residual / len(labels)

    result = minimize(
        objective,
        initial_weights,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * model_count,
        constraints={
            "type": "eq",
            "fun": lambda weights: float(np.sum(weights) - 1.0),
            "jac": lambda weights: np.ones_like(weights),
        },
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Ensemble weight optimization failed: {result.message}")

    optimized_weights = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
    optimized_weights /= optimized_weights.sum()
    if not np.isclose(optimized_weights.sum(), 1.0, atol=1e-8):
        raise RuntimeError("Optimized ensemble weights do not sum to one.")
    return {
        model_name: float(weight)
        for model_name, weight in zip(model_names, optimized_weights, strict=True)
    }


def _read_training_partition(
    disease: str,
    dataset_dir: Path,
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any], dict[str, Any]]:
    dataset = load_dataset(dataset_dir / f"{disease}.csv")
    split_path = results_dir / disease / "data_split.json"
    training_path = results_dir / disease / "training.json"
    if not split_path.is_file() or not training_path.is_file():
        raise FileNotFoundError(
            f"Missing training artifacts for {disease}; run src.train first."
        )

    split_report = json.loads(split_path.read_text(encoding="utf-8"))
    training_report = json.loads(training_path.read_text(encoding="utf-8"))
    train_indices = np.asarray(split_report["training_indices"], dtype=int)
    test_indices = np.asarray(split_report["test_indices"], dtype=int)
    all_indices = set(range(len(dataset.target)))
    if (
        len(np.unique(train_indices)) != len(train_indices)
        or len(np.unique(test_indices)) != len(test_indices)
        or set(train_indices) & set(test_indices)
        or set(train_indices) | set(test_indices) != all_indices
    ):
        raise ValueError(f"Saved row split for {disease} is invalid or incomplete.")

    x_train = dataset.features.iloc[train_indices]
    y_train = dataset.target.iloc[train_indices]
    if len(x_train) != training_report["training_samples"]:
        raise ValueError(f"Training row count for {disease} does not match its report.")
    return x_train, y_train, split_report, training_report


def _write_cv_comparison(
    disease: str,
    training_report: dict[str, Any],
    results_path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for model_name in BASE_MODEL_NAMES:
        metrics = training_report["models"][model_name]["cross_validation"]["metrics"]
        for metric_name in CV_METRICS:
            metric = metrics[metric_name]
            rows.append(
                {
                    "disease": disease,
                    "model": model_name,
                    "metric": metric_name,
                    "mean": metric["mean"],
                    "std": metric["std"],
                    "status": "training-only stratified CV",
                }
            )
    for metric_name in CV_METRICS:
        rows.append(
            {
                "disease": disease,
                "model": "weighted_soft_voting",
                "metric": metric_name,
                "mean": "",
                "std": "",
                "status": (
                    "Not yet evaluated; weight fitting used training-only OOF data. "
                    "Final fair comparison is reserved for Phase 6."
                ),
            }
        )

    with results_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("disease", "model", "metric", "mean", "std", "status"),
        )
        writer.writeheader()
        writer.writerows(rows)


def compare_disease(
    disease: str,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    cv_folds: int = DEFAULT_CV_FOLDS,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit weights on training-only OOF probabilities and save an ensemble."""
    if disease not in FEATURE_SCHEMAS:
        raise ValueError(f"Unknown disease: {disease!r}")

    x_train, y_train, split_report, training_report = _read_training_partition(
        disease, dataset_dir, results_dir
    )
    if split_report["seed"] != seed or training_report["seed"] != seed:
        raise ValueError(
            f"Requested seed {seed} does not match the saved {disease} training split."
        )
    expected_folds = training_report["cross_validation"]["folds"]
    if cv_folds != expected_folds:
        raise ValueError(
            f"Requested {cv_folds} folds but {disease} models were trained with "
            f"{expected_folds}; rerun training with matching settings."
        )

    disease_model_dir = model_dir / disease
    disease_results_dir = results_dir / disease
    ensemble_path = disease_model_dir / "weighted_ensemble.joblib"
    weights_path = disease_results_dir / "ensemble.json"
    comparison_path = disease_results_dir / "cross_validation_comparison.csv"
    existing = [
        path for path in (ensemble_path, weights_path, comparison_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "Ensemble artifacts already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    oof_predictions: list[np.ndarray] = []
    fitted_estimators: dict[str, Any] = {}
    for model_name in BASE_MODEL_NAMES:
        model_path = disease_model_dir / f"{model_name}.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing fitted base model: {model_path}")
        estimator = joblib.load(model_path)
        classes = np.asarray(estimator.classes_)
        if not np.array_equal(classes, np.asarray([0, 1])):
            raise ValueError(
                f"{disease}/{model_name} class order must be [0, 1]; "
                f"got {classes.tolist()}."
            )
        oof_probabilities = cross_val_predict(
            estimator,
            x_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )
        if oof_probabilities.shape != (len(y_train), 2):
            raise ValueError(
                f"{disease}/{model_name} returned invalid OOF shape "
                f"{oof_probabilities.shape}."
            )
        oof_predictions.append(np.asarray(oof_probabilities[:, 1], dtype=float))
        fitted_estimators[model_name] = estimator

    oof_matrix = np.column_stack(oof_predictions)
    weights = derive_ensemble_weights(oof_matrix, y_train.to_numpy(), BASE_MODEL_NAMES)
    weighted_oof = oof_matrix @ np.asarray([weights[name] for name in BASE_MODEL_NAMES])
    fitting_objective = float(np.mean((weighted_oof - y_train.to_numpy()) ** 2))
    ensemble = WeightedSoftVotingClassifier(fitted_estimators, weights)

    disease_model_dir.mkdir(parents=True, exist_ok=True)
    disease_results_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble, ensemble_path)
    ensemble_report = {
        "disease": disease,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ensemble_type": "weighted soft voting",
        "probability_formula": "P(y=1|x) = sum_i(weight_i * P_i(y=1|x))",
        "positive_class": 1,
        "base_models": list(BASE_MODEL_NAMES),
        "weights": weights,
        "weight_constraints": {
            "non_negative": all(weight >= 0 for weight in weights.values()),
            "sum_to_one": float(sum(weights.values())),
        },
        "weight_selection": {
            "method": "constrained Brier-score minimization on training OOF probabilities",
            "cv_strategy": "StratifiedKFold(shuffle=True)",
            "folds": cv_folds,
            "seed": seed,
            "training_samples": int(len(y_train)),
            "held_out_test_used": False,
            "training_oof_brier_objective": fitting_objective,
            "objective_interpretation": (
                "Optimization objective on the same OOF predictions used to choose "
                "weights; not an unbiased generalization estimate."
            ),
        },
        "saved_model": str(ensemble_path.relative_to(PROJECT_ROOT)),
        "held_out_test_evaluation": "Not yet evaluated",
    }
    weights_path.write_text(json.dumps(ensemble_report, indent=2), encoding="utf-8")
    _write_cv_comparison(disease, training_report, comparison_path)
    return ensemble_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease", choices=tuple(FEATURE_SCHEMAS))
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    diseases = [args.disease] if args.disease else list(FEATURE_SCHEMAS)
    try:
        for disease in diseases:
            report = compare_disease(
                disease,
                cv_folds=args.cv_folds,
                seed=args.seed,
                overwrite=args.overwrite,
            )
            print(f"{disease}: training-only weighted ensemble created")
            for model_name, weight in report["weights"].items():
                print(f"  {model_name}: weight={weight:.6f}")
            print(
                "  test performance: "
                f"{report['held_out_test_evaluation']}"
            )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Train individual baseline models with stratified, training-only CV."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from xgboost import XGBClassifier

from src.preprocessing import (
    FEATURE_SCHEMAS,
    build_training_pipeline,
    load_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_FOLDS = 5


def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    denominator = true_negative + false_positive
    return float(true_negative / denominator) if denominator else 0.0


def _scoring() -> dict[str, Any]:
    return {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall_sensitivity": make_scorer(recall_score, zero_division=0),
        "specificity": make_scorer(_specificity),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": make_scorer(roc_auc_score, response_method="predict_proba"),
        "average_precision": make_scorer(
            average_precision_score, response_method="predict_proba"
        ),
        "brier_score": make_scorer(
            brier_score_loss,
            response_method="predict_proba",
            greater_is_better=False,
        ),
    }


def build_estimators(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Return fixed, CPU-friendly baselines; no held-out data informs settings."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        ),
    }


def _class_distribution(target: pd.Series) -> dict[str, int]:
    counts = target.value_counts(sort=False)
    return {str(label): int(count) for label, count in counts.items()}


def _summarize_cv(cv_results: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for result_key, values in cv_results.items():
        if not result_key.startswith("test_"):
            continue
        metric_name = result_key.removeprefix("test_")
        scores = np.asarray(values, dtype=float)
        if metric_name == "brier_score":
            scores = -scores
        summary[metric_name] = {
            "mean": float(scores.mean()),
            "std": float(scores.std(ddof=1)),
            "fold_scores": [float(score) for score in scores],
        }
    return summary


def train_disease(
    disease: str,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    cv_folds: int = DEFAULT_CV_FOLDS,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Split once, cross-validate on training data, and save fitted pipelines."""
    if disease not in FEATURE_SCHEMAS:
        raise ValueError(f"Unknown disease: {disease!r}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")

    dataset = load_dataset(dataset_dir / f"{disease}.csv")
    all_indices = np.arange(len(dataset.target))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=seed,
        stratify=dataset.target,
    )
    x_train = dataset.features.iloc[train_indices]
    y_train = dataset.target.iloc[train_indices]
    x_test = dataset.features.iloc[test_indices]
    y_test = dataset.target.iloc[test_indices]

    smallest_training_class = int(y_train.value_counts().min())
    if smallest_training_class < cv_folds:
        raise ValueError(
            f"{disease} has only {smallest_training_class} training samples in its "
            f"smallest class; cannot run {cv_folds}-fold stratified CV."
        )

    disease_model_dir = model_dir / disease
    disease_results_dir = results_dir / disease
    model_paths = {
        model_name: disease_model_dir / f"{model_name}.joblib"
        for model_name in build_estimators(seed)
    }
    report_path = disease_results_dir / "training.json"
    split_path = disease_results_dir / "data_split.json"
    existing = [
        path for path in [*model_paths.values(), report_path, split_path] if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "Training artifacts already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    disease_model_dir.mkdir(parents=True, exist_ok=True)
    disease_results_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    model_results: dict[str, Any] = {}

    for model_name, estimator in build_estimators(seed).items():
        pipeline = build_training_pipeline(estimator, x_train, disease)
        cv_results = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring=_scoring(),
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        cv_summary = _summarize_cv(cv_results)
        pipeline.fit(x_train, y_train)
        classes = pipeline.named_steps["model"].classes_.tolist()
        if classes != [0, 1]:
            raise ValueError(
                f"{disease}/{model_name} produced unexpected class order: {classes}"
            )
        joblib.dump(pipeline, model_paths[model_name])
        try:
            saved_model_path = model_paths[model_name].relative_to(PROJECT_ROOT)
        except ValueError:
            saved_model_path = model_paths[model_name].resolve()
        model_results[model_name] = {
            "saved_model": str(saved_model_path),
            "model_parameters": {
                key: value
                for key, value in estimator.get_params(deep=False).items()
                if key
                in {
                    "C",
                    "max_iter",
                    "solver",
                    "n_estimators",
                    "max_depth",
                    "min_samples_leaf",
                    "learning_rate",
                    "subsample",
                    "colsample_bytree",
                    "reg_lambda",
                    "objective",
                    "eval_metric",
                    "tree_method",
                    "random_state",
                    "n_jobs",
                }
            },
            "classes": classes,
            "cross_validation": {
                "folds": cv_folds,
                "strategy": "StratifiedKFold(shuffle=True)",
                "scoring_partition": "training split only",
                "metrics": cv_summary,
            },
        }

    split_report = {
        "dataset": disease,
        "seed": seed,
        "test_size_fraction": test_size,
        "split_method": "train_test_split(stratify=target)",
        "row_index_reference": "zero-based row positions after Phase 3 cleanup",
        "training_indices": [int(index) for index in train_indices],
        "test_indices": [int(index) for index in test_indices],
        "training_class_distribution": _class_distribution(y_train),
        "test_class_distribution": _class_distribution(y_test),
        "note": "Test labels are saved only to reproduce the partition; they were not used for CV or fitting.",
    }
    training_report = {
        "dataset": disease,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "training_samples": int(len(x_train)),
        "test_samples_reserved": int(len(x_test)),
        "feature_count_before_encoding": int(x_train.shape[1]),
        "training_class_distribution": _class_distribution(y_train),
        "test_class_distribution_reserved": _class_distribution(y_test),
        "cleaning_report": dataset.report,
        "cross_validation": {
            "folds": cv_folds,
            "stratified": True,
            "shuffle": True,
            "seed": seed,
            "preprocessing_refit_inside_each_fold": True,
        },
        "software_versions": {
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "models": model_results,
        "test_evaluation": "Not yet evaluated",
    }
    report_path.write_text(json.dumps(training_report, indent=2), encoding="utf-8")
    split_path.write_text(json.dumps(split_report, indent=2), encoding="utf-8")
    return training_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--disease",
        choices=tuple(FEATURE_SCHEMAS),
        help="Train only one module; by default train all three.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    diseases = [args.disease] if args.disease else list(FEATURE_SCHEMAS)
    try:
        for disease in diseases:
            report = train_disease(
                disease,
                seed=args.seed,
                test_size=args.test_size,
                cv_folds=args.cv_folds,
                overwrite=args.overwrite,
            )
            print(
                f"{disease}: trained {len(report['models'])} models on "
                f"{report['training_samples']} rows; held out "
                f"{report['test_samples_reserved']} rows."
            )
            for model_name, model_report in report["models"].items():
                roc_auc = model_report["cross_validation"]["metrics"]["roc_auc"]
                print(
                    f"  {model_name}: training CV ROC-AUC "
                    f"{roc_auc['mean']:.4f} +/- {roc_auc['std']:.4f}"
                )
    except (FileExistsError, FileNotFoundError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
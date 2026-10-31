"""Quantify SHAP stability under numeric perturbations and method agreement."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

from src.explain import (
    _create_explainer,
    normalize_shap_outputs,
    select_model_from_cv,
)
from src.preprocessing import (
    FEATURE_SCHEMAS,
    load_dataset,
    transformed_feature_mapping,
)


matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_PERTURBATION_MAGNITUDES = (0.05, 0.10)
DEFAULT_REPETITIONS = 10
DEFAULT_SAMPLE_SIZE = 30
DEFAULT_PERMUTATION_REPEATS = 10
RANDOM_SEED = 42


def perturb_numeric_features(
    features: pd.DataFrame,
    training_features: pd.DataFrame,
    numeric_features: Sequence[str],
    magnitude: float,
    random_generator: np.random.Generator,
) -> pd.DataFrame:
    """Perturb numeric values by training SD multiples and clip to train ranges."""
    if magnitude < 0 or not np.isfinite(magnitude):
        raise ValueError("Perturbation magnitude must be finite and non-negative.")
    perturbed = features.copy(deep=True)
    for column in numeric_features:
        training_values = pd.to_numeric(training_features[column], errors="coerce")
        observed_training = training_values.dropna()
        if observed_training.empty:
            continue
        standard_deviation = float(observed_training.std(ddof=0))
        if not np.isfinite(standard_deviation) or standard_deviation == 0:
            continue

        test_values = pd.to_numeric(perturbed[column], errors="coerce").astype(float)
        observed_mask = test_values.notna()
        if not observed_mask.any():
            perturbed[column] = test_values
            continue
        noise = random_generator.normal(
            loc=0.0,
            scale=magnitude * standard_deviation,
            size=int(observed_mask.sum()),
        )
        lower_bound = float(observed_training.min())
        upper_bound = float(observed_training.max())
        test_values.loc[observed_mask] = np.clip(
            test_values.loc[observed_mask].to_numpy() + noise,
            lower_bound,
            upper_bound,
        )
        perturbed[column] = test_values
    return perturbed


def aggregate_source_importance(
    shap_values: np.ndarray,
    transformed_feature_names: Sequence[str],
    transformed_to_source: dict[str, str],
) -> tuple[list[str], np.ndarray]:
    """Sum absolute encoded-feature contributions within each original feature."""
    values = np.asarray(shap_values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(transformed_feature_names):
        raise ValueError("SHAP matrix and transformed feature names do not align.")
    source_columns: dict[str, list[int]] = {}
    for index, transformed_name in enumerate(transformed_feature_names):
        source_name = transformed_to_source.get(transformed_name, transformed_name)
        source_columns.setdefault(source_name, []).append(index)
    source_names = list(source_columns)
    importance = np.column_stack(
        [np.abs(values[:, source_columns[name]]).sum(axis=1) for name in source_names]
    )
    return source_names, importance


def rank_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    """Return Spearman correlation, defining exact constant ties as stable."""
    first_values = np.asarray(first, dtype=float)
    second_values = np.asarray(second, dtype=float)
    if first_values.shape != second_values.shape or first_values.ndim != 1:
        raise ValueError("Rank-correlation inputs must be same-length 1D arrays.")
    if np.allclose(first_values, second_values):
        return 1.0
    if np.ptp(first_values) == 0 or np.ptp(second_values) == 0:
        return 0.0
    statistic = float(spearmanr(first_values, second_values).statistic)
    if np.isfinite(statistic):
        return statistic
    return 0.0


def top_k_jaccard(
    first: Sequence[float],
    second: Sequence[float],
    k: int = 5,
) -> float:
    """Compare top-k index sets from two original-feature importance vectors."""
    first_values = np.asarray(first, dtype=float)
    second_values = np.asarray(second, dtype=float)
    if first_values.shape != second_values.shape or first_values.ndim != 1:
        raise ValueError("Top-k inputs must be same-length 1D arrays.")
    if k < 1:
        raise ValueError("k must be positive.")
    top_count = min(k, len(first_values))
    first_top = set(np.argsort(-first_values, kind="stable")[:top_count])
    second_top = set(np.argsort(-second_values, kind="stable")[:top_count])
    union = first_top | second_top
    return float(len(first_top & second_top) / len(union)) if union else 1.0


def _read_model_data(
    disease: str,
    dataset_dir: Path,
    model_dir: Path,
    results_dir: Path,
) -> tuple[Any, Any, Any, Any, dict[str, Any], str]:
    dataset = load_dataset(dataset_dir / f"{disease}.csv")
    disease_results_dir = results_dir / disease
    training_report = json.loads(
        (disease_results_dir / "training.json").read_text(encoding="utf-8")
    )
    split_report = json.loads(
        (disease_results_dir / "data_split.json").read_text(encoding="utf-8")
    )
    selected_model, _ = select_model_from_cv(training_report)
    model_path = model_dir / disease / f"{selected_model}.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing CV-selected model: {model_path}")

    train_indices = np.asarray(split_report["training_indices"], dtype=int)
    test_indices = np.asarray(split_report["test_indices"], dtype=int)
    all_indices = set(range(len(dataset.target)))
    if (
        set(train_indices) & set(test_indices)
        or set(train_indices) | set(test_indices) != all_indices
    ):
        raise ValueError(f"Saved row split for {disease} is invalid or incomplete.")
    x_train = dataset.features.iloc[train_indices]
    y_test = dataset.target.iloc[test_indices]
    x_test = dataset.features.iloc[test_indices]
    return (
        joblib.load(model_path),
        x_train,
        (x_test, y_test, test_indices),
        dataset,
        training_report,
        selected_model,
    )


def _compute_shap_values(
    explainer: Any,
    estimator: Any,
    features: pd.DataFrame,
    preprocessor: Any,
) -> tuple[np.ndarray, np.ndarray]:
    transformed = np.asarray(preprocessor.transform(features), dtype=float)
    raw_values = explainer.shap_values(transformed)
    values, base_values = normalize_shap_outputs(
        raw_values,
        explainer.expected_value,
        np.asarray(estimator.classes_),
        row_count=len(features),
        feature_count=transformed.shape[1],
    )
    return values, transformed


def _summarize_stability(records: pd.DataFrame, magnitudes: Sequence[float]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for magnitude in magnitudes:
        subset = records.loc[records["magnitude_std"] == magnitude]
        repetition_means = subset.groupby("repetition").agg(
            spearman_mean=("spearman_rank_correlation", "mean"),
            jaccard_mean=("top5_jaccard", "mean"),
        )
        summaries[str(magnitude)] = {
            "magnitude_in_training_standard_deviations": float(magnitude),
            "comparisons": int(len(subset)),
            "spearman_rank_correlation_mean": float(
                subset["spearman_rank_correlation"].mean()
            ),
            "spearman_rank_correlation_std": float(
                subset["spearman_rank_correlation"].std(ddof=1)
                if len(subset) > 1
                else 0.0
            ),
            "top5_jaccard_mean": float(subset["top5_jaccard"].mean()),
            "top5_jaccard_std": float(
                subset["top5_jaccard"].std(ddof=1) if len(subset) > 1 else 0.0
            ),
            "per_repetition_means": [
                {
                    "repetition": int(repetition),
                    "spearman_rank_correlation_mean": float(row["spearman_mean"]),
                    "top5_jaccard_mean": float(row["jaccard_mean"]),
                }
                for repetition, row in repetition_means.iterrows()
            ],
        }
    return summaries


def _save_stability_plot(
    summaries: dict[str, Any],
    output_path: Path,
) -> None:
    magnitudes = [float(key) for key in summaries]
    spearman_means = [
        summaries[key]["spearman_rank_correlation_mean"] for key in summaries
    ]
    spearman_stds = [
        summaries[key]["spearman_rank_correlation_std"] for key in summaries
    ]
    jaccard_means = [summaries[key]["top5_jaccard_mean"] for key in summaries]
    jaccard_stds = [summaries[key]["top5_jaccard_std"] for key in summaries]

    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.errorbar(
        magnitudes,
        spearman_means,
        yerr=spearman_stds,
        marker="o",
        capsize=4,
        label="Spearman rank correlation",
    )
    axis.errorbar(
        magnitudes,
        jaccard_means,
        yerr=jaccard_stds,
        marker="s",
        capsize=4,
        label="Top-5 Jaccard similarity",
    )
    axis.set(
        xlabel="Perturbation magnitude (multiples of training SD)",
        ylabel="Stability / overlap",
        ylim=(-0.05, 1.05),
        title="SHAP explanation stability under numeric perturbations",
    )
    axis.legend(loc="best")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def evaluate_reliability(
    disease: str,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    magnitudes: Sequence[float] = DEFAULT_PERTURBATION_MAGNITUDES,
    repetitions: int = DEFAULT_REPETITIONS,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    permutation_repeats: int = DEFAULT_PERMUTATION_REPEATS,
    seed: int = RANDOM_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run perturbation stability and SHAP/permutation agreement experiments."""
    if disease not in FEATURE_SCHEMAS:
        raise ValueError(f"Unknown disease: {disease!r}")
    if not magnitudes or any(not np.isfinite(value) or value <= 0 for value in magnitudes):
        raise ValueError("All perturbation magnitudes must be finite and positive.")
    if repetitions < 1 or sample_size < 1 or permutation_repeats < 1:
        raise ValueError("Repetition and sample counts must be positive.")

    model, x_train, test_data, dataset, training_report, selected_model = _read_model_data(
        disease, dataset_dir, model_dir, results_dir
    )
    x_test, y_test, test_indices = test_data
    disease_results_dir = results_dir / disease
    output_dir = disease_results_dir / "reliability"
    report_path = output_dir / "reliability.json"
    stability_path = output_dir / "perturbation_stability.csv"
    permutation_path = output_dir / "shap_permutation_importance.csv"
    plot_path = output_dir / "stability_by_magnitude.png"
    existing = [
        path
        for path in (report_path, stability_path, permutation_path, plot_path)
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "Reliability artifacts already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    transformed_feature_names = preprocessor.get_feature_names_out().tolist()
    transformed_to_source = transformed_feature_mapping(preprocessor)
    background_size = min(50, len(x_train))
    background_frame = x_train.sample(n=background_size, random_state=seed)
    background = np.asarray(preprocessor.transform(background_frame), dtype=float)
    explainer, output_space = _create_explainer(
        selected_model, estimator, background
    )

    original_shap, _ = _compute_shap_values(
        explainer, estimator, x_test, preprocessor
    )
    source_names, original_importance = aggregate_source_importance(
        original_shap,
        transformed_feature_names,
        transformed_to_source,
    )
    rng = np.random.default_rng(seed)
    selected_positions = np.sort(
        rng.choice(len(x_test), size=min(sample_size, len(x_test)), replace=False)
    )
    selected_test = x_test.iloc[selected_positions]
    original_selected_importance = original_importance[selected_positions]
    numeric_features = list(FEATURE_SCHEMAS[disease].numeric)
    stability_records: list[dict[str, Any]] = []

    for magnitude_index, magnitude in enumerate(magnitudes):
        for repetition in range(repetitions):
            perturb_rng = np.random.default_rng(
                seed + magnitude_index * 100_000 + repetition
            )
            perturbed_features = perturb_numeric_features(
                selected_test,
                x_train,
                numeric_features,
                float(magnitude),
                perturb_rng,
            )
            perturbed_shap, _ = _compute_shap_values(
                explainer, estimator, perturbed_features, preprocessor
            )
            _, perturbed_importance = aggregate_source_importance(
                perturbed_shap,
                transformed_feature_names,
                transformed_to_source,
            )
            for row_position, original_row_index in enumerate(selected_positions):
                stability_records.append(
                    {
                        "disease": disease,
                        "magnitude_std": float(magnitude),
                        "repetition": repetition,
                        "test_row_index": int(test_indices[original_row_index]),
                        "spearman_rank_correlation": rank_correlation(
                            original_selected_importance[row_position],
                            perturbed_importance[row_position],
                        ),
                        "top5_jaccard": top_k_jaccard(
                            original_selected_importance[row_position],
                            perturbed_importance[row_position],
                            k=5,
                        ),
                    }
                )

    stability_frame = pd.DataFrame(stability_records)
    stability_summary = _summarize_stability(stability_frame, magnitudes)

    permutation_result = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=permutation_repeats,
        random_state=seed,
        n_jobs=1,
    )
    permutation_importances = {
        feature: index for index, feature in enumerate(x_test.columns)
    }
    shap_source_importance = {
        feature: float(original_importance[:, index].mean())
        for index, feature in enumerate(source_names)
    }
    shap_vector = np.asarray(
        [shap_source_importance[feature] for feature in x_test.columns], dtype=float
    )
    permutation_vector = np.asarray(
        [
            permutation_result.importances_mean[permutation_importances[feature]]
            for feature in x_test.columns
        ],
        dtype=float,
    )
    shap_permutation_correlation = rank_correlation(
        shap_vector, permutation_vector
    )
    shap_ranks = pd.Series(shap_vector).rank(ascending=False, method="average")
    permutation_ranks = pd.Series(permutation_vector).rank(
        ascending=False, method="average"
    )
    permutation_frame = pd.DataFrame(
        {
            "feature": list(x_test.columns),
            "mean_absolute_shap": shap_vector,
            "permutation_roc_auc_importance_mean": permutation_vector,
            "permutation_roc_auc_importance_std": [
                float(permutation_result.importances_std[index])
                for index in range(len(x_test.columns))
            ],
            "shap_rank": shap_ranks,
            "permutation_rank": permutation_ranks,
        }
    ).sort_values("shap_rank")

    output_dir.mkdir(parents=True, exist_ok=True)
    stability_frame.to_csv(stability_path, index=False)
    permutation_frame.to_csv(permutation_path, index=False)
    _save_stability_plot(stability_summary, plot_path)
    report = {
        "disease": disease,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected_model,
        "model_output_space": output_space,
        "selection_criterion": "highest mean 5-fold training CV ROC-AUC; ties use lower standard deviation",
        "test_records_used": int(len(x_test)),
        "numerical_features_perturbed": numeric_features,
        "perturbation_experiment": {
            "method": "independent Gaussian noise applied to numeric source fields; categorical values unchanged",
            "magnitude_unit": "multiples of the training-partition standard deviation",
            "magnitudes": [float(magnitude) for magnitude in magnitudes],
            "repetitions_per_magnitude": int(repetitions),
            "records_per_repetition": int(len(selected_positions)),
            "random_seed": int(seed),
            "clipping": "perturbed numeric values clipped to each feature's observed training min/max; missing values stay missing",
            "ranking_unit": "original source features, summing absolute SHAP values across one-hot columns per row",
            "top_k": 5,
            "stability_by_magnitude": stability_summary,
        },
        "shap_permutation_agreement": {
            "permutation_importance_metric": "ROC-AUC",
            "permutation_repeats": int(permutation_repeats),
            "shap_global_importance": "mean absolute source-feature SHAP aggregated over the full held-out test partition",
            "spearman_rank_correlation": shap_permutation_correlation,
            "features": permutation_frame.to_dict(orient="records"),
        },
        "artifacts": {
            "perturbation_stability_csv": str(stability_path),
            "shap_permutation_importance_csv": str(permutation_path),
            "stability_plot": str(plot_path),
        },
        "interpretation_limitations": (
            "These metrics describe explanation stability and agreement under the "
            "specified perturbation and permutation procedures. They do not prove "
            "clinical correctness, fairness, causality, or medical reliability."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease", choices=tuple(FEATURE_SCHEMAS))
    parser.add_argument("--magnitudes", nargs="+", type=float, default=list(DEFAULT_PERTURBATION_MAGNITUDES))
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--permutation-repeats", type=int, default=DEFAULT_PERMUTATION_REPEATS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    diseases = [args.disease] if args.disease else list(FEATURE_SCHEMAS)
    try:
        for disease in diseases:
            report = evaluate_reliability(
                disease,
                magnitudes=args.magnitudes,
                repetitions=args.repetitions,
                sample_size=args.sample_size,
                permutation_repeats=args.permutation_repeats,
                seed=args.seed,
                overwrite=args.overwrite,
            )
            print(
                f"{disease}: SHAP/permutation Spearman="
                f"{report['shap_permutation_agreement']['spearman_rank_correlation']:.3f}"
            )
            for magnitude, summary in report["perturbation_experiment"]["stability_by_magnitude"].items():
                print(
                    f"  {magnitude} SD: rank rho="
                    f"{summary['spearman_rank_correlation_mean']:.3f} +/- "
                    f"{summary['spearman_rank_correlation_std']:.3f}; "
                    f"top-5 Jaccard={summary['top5_jaccard_mean']:.3f} +/- "
                    f"{summary['top5_jaccard_std']:.3f}"
                )
    except (FileExistsError, FileNotFoundError, TypeError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
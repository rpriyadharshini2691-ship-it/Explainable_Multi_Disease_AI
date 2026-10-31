import numpy as np
import pandas as pd

from src.train import train_disease


def _synthetic_diabetes_frame(row_count=48):
    random_generator = np.random.default_rng(42)
    target = np.tile([0, 1], row_count // 2)
    feature_values = {
        "age": random_generator.integers(18, 66, size=row_count),
    }
    categorical_features = (
        "gender",
        "polyuria",
        "polydipsia",
        "sudden_weight_loss",
        "weakness",
        "polyphagia",
        "genital_thrush",
        "visual_blurring",
        "itching",
        "irritability",
        "delayed_healing",
        "partial_paresis",
        "muscle_stiffness",
        "alopecia",
        "obesity",
    )
    for feature in categorical_features:
        choices = ["Female", "Male"] if feature == "gender" else ["No", "Yes"]
        feature_values[feature] = random_generator.choice(choices, size=row_count)
    frame = pd.DataFrame(feature_values)
    frame["target"] = target
    return frame


def test_train_disease_runs_stratified_cv_and_saves_models(tmp_path):
    dataset_dir = tmp_path / "datasets"
    model_dir = tmp_path / "models"
    results_dir = tmp_path / "results"
    dataset_dir.mkdir()
    _synthetic_diabetes_frame().to_csv(dataset_dir / "diabetes.csv", index=False)

    report = train_disease(
        "diabetes",
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        results_dir=results_dir,
        test_size=0.25,
        cv_folds=2,
    )

    assert report["training_samples"] == 36
    assert report["test_samples_reserved"] == 12
    assert report["cross_validation"]["preprocessing_refit_inside_each_fold"]
    assert set(report["models"]) == {
        "logistic_regression",
        "random_forest",
        "xgboost",
    }
    assert report["test_evaluation"] == "Not yet evaluated"
    for model_report in report["models"].values():
        assert model_report["cross_validation"]["folds"] == 2
        assert "roc_auc" in model_report["cross_validation"]["metrics"]
        assert model_report["saved_model"].endswith(".joblib")
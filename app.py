"""Streamlit interface for the multi-disease explainable AI prototype."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

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

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "datasets"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

DISEASE_LABELS = {
    "diabetes": "Early-stage diabetes",
    "heart": "Heart disease",
    "kidney": "Chronic kidney disease",
}

FEATURE_LABELS = {
    "age": ("Age", "years"),
    "gender": ("Gender", ""),
    "polyuria": ("Frequent urination", ""),
    "polydipsia": ("Excessive thirst", ""),
    "sudden_weight_loss": ("Sudden weight loss", ""),
    "weakness": ("Weakness", ""),
    "polyphagia": ("Excessive hunger", ""),
    "genital_thrush": ("Genital thrush", ""),
    "visual_blurring": ("Blurred vision", ""),
    "itching": ("Itching", ""),
    "irritability": ("Irritability", ""),
    "delayed_healing": ("Delayed healing", ""),
    "partial_paresis": ("Partial paresis", ""),
    "muscle_stiffness": ("Muscle stiffness", ""),
    "alopecia": ("Hair loss", ""),
    "obesity": ("Obesity indicator", ""),
    "sex": ("Sex", ""),
    "cp": ("Chest-pain type", "UCI code"),
    "trestbps": ("Resting blood pressure", "mm Hg"),
    "chol": ("Serum cholesterol", "mg/dL"),
    "fbs": ("Fasting blood sugar above 120", "UCI code"),
    "restecg": ("Resting ECG result", "UCI code"),
    "thalach": ("Maximum heart rate", "beats/min"),
    "exang": ("Exercise-induced angina", "UCI code"),
    "oldpeak": ("Exercise ST depression", "relative to rest"),
    "slope": ("Peak exercise ST slope", "UCI code"),
    "ca": ("Major vessels visible by fluoroscopy", "count"),
    "thal": ("Thalassemia test result", "UCI code"),
    "bp": ("Blood pressure", "mm Hg"),
    "sg": ("Urine specific gravity", ""),
    "al": ("Urine albumin level", "UCI scale"),
    "su": ("Urine sugar level", "UCI scale"),
    "rbc": ("Red blood cells", ""),
    "pc": ("Pus cells", ""),
    "pcc": ("Pus cell clumps", ""),
    "ba": ("Bacteria in urine", ""),
    "bgr": ("Random blood glucose", "mg/dL"),
    "bu": ("Blood urea", "mg/dL"),
    "sc": ("Serum creatinine", "mg/dL"),
    "sod": ("Serum sodium", "mEq/L"),
    "pot": ("Serum potassium", "mEq/L"),
    "hemo": ("Hemoglobin", "g/dL"),
    "pcv": ("Packed cell volume", "%"),
    "wbcc": ("White blood cell count", "cells/cumm"),
    "rbcc": ("Red blood cell count", "millions/cmm"),
    "htn": ("Hypertension", ""),
    "dm": ("Diabetes mellitus indicator", ""),
    "cad": ("Coronary artery disease indicator", ""),
    "appet": ("Appetite", ""),
    "pe": ("Pedal edema", ""),
    "ane": ("Anemia indicator", ""),
}

CATEGORY_LABELS = {
    ("heart", "sex"): {0: "Female", 1: "Male"},
    ("heart", "cp"): {
        1: "Typical angina",
        2: "Atypical angina",
        3: "Non-anginal pain",
        4: "Asymptomatic",
    },
    ("heart", "fbs"): {0: "No", 1: "Yes"},
    ("heart", "restecg"): {
        0: "Normal",
        1: "ST-T wave abnormality",
        2: "Probable/definite left ventricular hypertrophy",
    },
    ("heart", "exang"): {0: "No", 1: "Yes"},
    ("heart", "slope"): {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    ("heart", "thal"): {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
    ("diabetes", "gender"): {"Female": "Female", "Male": "Male"},
    ("kidney", "rbc"): {"normal": "Normal", "abnormal": "Abnormal"},
    ("kidney", "pc"): {"normal": "Normal", "abnormal": "Abnormal"},
    ("kidney", "pcc"): {"present": "Present", "notpresent": "Not present"},
    ("kidney", "ba"): {"present": "Present", "notpresent": "Not present"},
    ("kidney", "htn"): {"yes": "Yes", "no": "No"},
    ("kidney", "dm"): {"yes": "Yes", "no": "No"},
    ("kidney", "cad"): {"yes": "Yes", "no": "No"},
    ("kidney", "appet"): {"good": "Good", "poor": "Poor"},
    ("kidney", "pe"): {"yes": "Yes", "no": "No"},
    ("kidney", "ane"): {"yes": "Yes", "no": "No"},
}

DISCLAIMER = (
    "This application is an academic AI research prototype. It does not diagnose "
    "diseases, recommend treatments, or replace professional medical assessment. "
    "Model outputs are not validated clinical risk estimates."
)


st.set_page_config(
    page_title="Explainable Multi-Disease AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root {
        --ink: #173238;
        --muted: #647d80;
        --paper: #f3f7f5;
        --line: #dce7e2;
        --teal: #16736b;
        --coral: #c76c4b;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }
    h1, h2, h3 { font-family: "Space Grotesk", sans-serif; color: var(--ink); }
    [data-testid="stSidebar"] { background: #e8f0ed; border-right: 1px solid var(--line); }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid var(--line); padding: 14px; border-radius: 6px; }
    div[data-testid="stForm"] { background: #ffffff; border: 1px solid var(--line); padding: 20px; border-radius: 6px; }
    .eyebrow { color: var(--teal); font-weight: 700; text-transform: uppercase; font-size: 0.76rem; letter-spacing: 0.06em; }
    .clinical-note { border-left: 4px solid var(--coral); background: #fff8f4; padding: 12px 16px; border-radius: 3px; color: #573b31; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model and explanation background...")
def load_disease_bundle(disease: str) -> dict[str, Any]:
    dataset = load_dataset(DATASET_DIR / f"{disease}.csv")
    training_report = json.loads(
        (RESULTS_DIR / disease / "training.json").read_text(encoding="utf-8")
    )
    split_report = json.loads(
        (RESULTS_DIR / disease / "data_split.json").read_text(encoding="utf-8")
    )
    selected_model, selected_cv = select_model_from_cv(training_report)
    model = joblib.load(MODEL_DIR / disease / f"{selected_model}.joblib")
    x_train = dataset.features.iloc[split_report["training_indices"]]
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    background_frame = x_train.sample(n=min(50, len(x_train)), random_state=42)
    background = np.asarray(preprocessor.transform(background_frame), dtype=float)
    explainer, output_space = _create_explainer(selected_model, estimator, background)
    return {
        "dataset": dataset,
        "training_report": training_report,
        "selected_model": selected_model,
        "selected_cv": selected_cv,
        "model": model,
        "preprocessor": preprocessor,
        "estimator": estimator,
        "explainer": explainer,
        "output_space": output_space,
        "feature_mapping": transformed_feature_mapping(preprocessor),
        "training_features": x_train,
    }


def _format_category(disease: str, feature: str, value: Any) -> str:
    value_map = CATEGORY_LABELS.get((disease, feature), {})
    if value in value_map:
        return value_map[value]
    if isinstance(value, (float, np.floating)):
        return f"{value:g}"
    return str(value)


def _feature_input(
    disease: str,
    feature: str,
    training_features: pd.DataFrame,
) -> Any:
    label, unit = FEATURE_LABELS.get((feature), (feature.replace("_", " ").title(), ""))
    full_label = f"{label} ({unit})" if unit else label
    numeric_features = FEATURE_SCHEMAS[disease].numeric
    if feature in numeric_features:
        values = pd.to_numeric(training_features[feature], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"No usable training range exists for {feature}.")
        minimum = float(values.min())
        maximum = float(values.max())
        default = float(values.median())
        integer_input = pd.api.types.is_integer_dtype(training_features[feature].dtype)
        if minimum == maximum:
            minimum -= 1.0
            maximum += 1.0
        return st.number_input(
            full_label,
            min_value=minimum,
            max_value=maximum,
            value=float(np.clip(default, minimum, maximum)),
            step=1.0 if integer_input else max((maximum - minimum) / 100.0, 0.01),
            format="%.0f" if integer_input else "%.2f",
            key=f"{disease}_{feature}",
        )

    observed_values = training_features[feature].dropna().drop_duplicates().tolist()
    if not observed_values:
        raise ValueError(f"No usable training categories exist for {feature}.")
    observed_values = sorted(observed_values, key=lambda value: str(value))
    return st.selectbox(
        full_label,
        options=observed_values,
        format_func=lambda value: _format_category(disease, feature, value),
        key=f"{disease}_{feature}",
    )


def _predict_and_explain(
    bundle: dict[str, Any],
    input_features: pd.DataFrame,
) -> tuple[float, pd.DataFrame, float, float]:
    model = bundle["model"]
    estimator = bundle["estimator"]
    classes = np.asarray(estimator.classes_)
    positive_indices = np.flatnonzero(classes == 1)
    if len(positive_indices) != 1:
        raise ValueError(f"Model positive-class labels are ambiguous: {classes.tolist()}")
    positive_probability = float(model.predict_proba(input_features)[0, positive_indices[0]])
    transformed = np.asarray(
        bundle["preprocessor"].transform(input_features), dtype=float
    )
    raw_values = bundle["explainer"].shap_values(transformed)
    values, base_values = normalize_shap_outputs(
        raw_values,
        bundle["explainer"].expected_value,
        classes,
        row_count=1,
        feature_count=transformed.shape[1],
    )

    grouped_values: dict[str, list[float]] = {}
    grouped_names: dict[str, list[str]] = {}
    transformed_names = bundle["preprocessor"].get_feature_names_out().tolist()
    for index, transformed_name in enumerate(transformed_names):
        source_name = bundle["feature_mapping"].get(transformed_name, transformed_name)
        grouped_values.setdefault(source_name, []).append(float(values[0, index]))
        grouped_names.setdefault(source_name, []).append(transformed_name)
    contribution_rows = []
    for source_name, contributions in grouped_values.items():
        contribution = float(sum(contributions))
        source_label = FEATURE_LABELS.get(
            source_name, (source_name.replace("_", " ").title(), "")
        )[0]
        contribution_rows.append(
            {
                "feature": source_label,
                "contribution": contribution,
                "absolute_contribution": abs(contribution),
                "direction": "Raises model output" if contribution >= 0 else "Lowers model output",
            }
        )
    contributions = pd.DataFrame(contribution_rows).sort_values(
        "absolute_contribution", ascending=False
    )
    return (
        positive_probability,
        contributions,
        float(base_values[0]),
        float(base_values[0] + values[0].sum()),
    )


def _render_home() -> None:
    st.markdown('<div class="eyebrow">Research prototype · CPU inference</div>', unsafe_allow_html=True)
    st.title("Explainable Multi-Disease AI")
    st.write(
        "Independent prediction modules for early-stage diabetes, Cleveland heart "
        "disease, and chronic kidney disease. Models are selected using training-only "
        "cross-validation; explanation and reliability artifacts are derived from "
        "the saved research runs."
    )
    manifest_path = DATASET_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        columns = st.columns(3)
        for column, disease in zip(columns, FEATURE_SCHEMAS, strict=True):
            entry = manifest["datasets"][disease]
            column.metric(
                DISEASE_LABELS[disease],
                f"{entry['saved_rows']:,} records",
                f"UCI dataset {entry['uci_id']}",
            )
    st.markdown(f'<div class="clinical-note">{DISCLAIMER}</div>', unsafe_allow_html=True)
    st.caption(
        "Submitted form values are used only for the current prediction and are not written to project files."
    )


def _render_prediction(disease: str, bundle: dict[str, Any]) -> None:
    st.subheader("Prediction")
    st.caption(
        f"Model: {bundle['selected_model'].replace('_', ' ').title()} · "
        f"training CV ROC-AUC {bundle['selected_cv']['mean']:.3f} "
        f"± {bundle['selected_cv']['std']:.3f}"
    )
    input_values: dict[str, Any] = {}
    with st.form(f"prediction_form_{disease}"):
        feature_columns = st.columns(2)
        for feature_index, feature in enumerate(bundle["dataset"].features.columns):
            with feature_columns[feature_index % 2]:
                input_values[feature] = _feature_input(
                    disease, feature, bundle["training_features"]
                )
        submitted = st.form_submit_button("Run model", type="primary")

    if submitted:
        input_features = pd.DataFrame([input_values])
        positive_score, contributions, base_value, model_output = _predict_and_explain(
            bundle, input_features
        )
        evaluation_path = RESULTS_DIR / disease / "evaluation.json"
        threshold = 0.5
        if evaluation_path.exists():
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            threshold = float(
                evaluation["threshold_selection"]["models"][bundle["selected_model"]][
                    "threshold"
                ]
            )
        predicted_class = int(positive_score >= threshold)

        result_columns = st.columns(3)
        result_columns[0].metric("Positive-class model score", f"{positive_score:.1%}")
        result_columns[1].metric("Model output class", "Positive" if predicted_class else "Negative")
        result_columns[2].metric("Training-selected threshold", f"{threshold:.3f}")
        st.caption(
            "The score is the model's positive-class output, not a validated clinical risk estimate. "
            f"SHAP explanations are in {bundle['output_space']} space."
        )
        st.markdown("#### Local feature contributions")
        st.caption(
            f"Base value {base_value:.3f} · reconstructed model output {model_output:.3f}. "
            "Positive contributions raise this model output; negative contributions lower it."
        )
        display_contributions = contributions.head(12).set_index("feature")
        st.bar_chart(
            display_contributions["contribution"],
            color="#16736b",
            horizontal=True,
            width="stretch",
        )
        st.dataframe(
            contributions[["feature", "contribution", "direction"]].head(12),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Feature contributions describe model behavior, not medical causation. "
            "No submitted values are persisted by this application."
        )


def _render_performance(disease: str, bundle: dict[str, Any]) -> None:
    st.subheader("Test-set performance")
    evaluation_path = RESULTS_DIR / disease / "evaluation.json"
    metrics_path = RESULTS_DIR / disease / "test_metrics.csv"
    if not evaluation_path.exists() or not metrics_path.exists():
        st.info("Evaluation artifacts are not available yet.")
        return

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    selected_metrics = evaluation["models"][bundle["selected_model"]]["test_metrics"]
    selected_columns = st.columns(2)
    for column, scenario, label in zip(
        selected_columns,
        ("default_0_5", "youden_threshold"),
        ("Default 0.5 threshold", "Training-only Youden threshold"),
        strict=True,
    ):
        metrics = selected_metrics[scenario]
        column.metric(f"{label} · accuracy", f"{metrics['accuracy']:.3f}")
        column.caption(
            f"Sensitivity {metrics['sensitivity']:.3f} · "
            f"specificity {metrics['specificity']:.3f} · "
            f"false-negative rate {metrics['false_negative_rate']:.3f} · "
            f"Brier {metrics['brier_score']:.3f}"
        )

    test_metrics = pd.read_csv(metrics_path)
    selected_table = test_metrics.loc[
        test_metrics["threshold_scenario"] == "default_0_5",
        [
            "model",
            "accuracy",
            "precision",
            "sensitivity",
            "specificity",
            "f1",
            "roc_auc",
            "average_precision",
            "brier_score",
        ],
    ]
    st.dataframe(selected_table, hide_index=True, width="stretch")
    st.caption("All models use the same held-out split. Results are experimental, not clinical validation.")

    comparison_path = RESULTS_DIR / disease / "cross_validation_comparison.csv"
    if comparison_path.exists():
        with st.expander("Training cross-validation comparison"):
            cv_comparison = pd.read_csv(comparison_path)
            st.dataframe(cv_comparison, hide_index=True, width="stretch")

    for plot_name, label in (
        ("test_roc_curves.png", "ROC curves"),
        ("test_precision_recall_curves.png", "Precision-recall curves"),
        ("test_calibration_curves.png", "Calibration curves"),
        ("test_confusion_matrices.png", "Confusion matrices"),
    ):
        plot_path = RESULTS_DIR / disease / "figures" / plot_name
        if plot_path.exists():
            st.markdown(f"#### {label}")
            st.image(str(plot_path), width="stretch")


def _render_explanations(disease: str, bundle: dict[str, Any]) -> None:
    st.subheader("Global SHAP explanation")
    explanation_path = RESULTS_DIR / disease / "shap" / "explanation_report.json"
    if not explanation_path.exists():
        st.info("SHAP artifacts are not available yet.")
        return
    explanation = json.loads(explanation_path.read_text(encoding="utf-8"))
    st.caption(
        f"{explanation['selected_model'].replace('_', ' ').title()} · "
        f"{explanation['model_output_space']} · "
        "held-out examples · global importance sums encoded contributions by source feature"
    )
    plot_path = RESULTS_DIR / disease / "shap" / "shap_summary_beeswarm.png"
    if plot_path.exists():
        st.image(str(plot_path), width="stretch")
    importance = pd.DataFrame(explanation["global_importance"])
    st.dataframe(importance.head(15), hide_index=True, width="stretch")
    global_plot = RESULTS_DIR / disease / "shap" / "shap_global_importance.png"
    if global_plot.exists():
        st.image(str(global_plot), width="stretch")
    local_plot = RESULTS_DIR / disease / "shap" / "shap_local_waterfall.png"
    if local_plot.exists():
        st.markdown("#### Saved held-out local explanation")
        st.image(str(local_plot), width="stretch")
    st.caption(explanation["interpretation_limitations"])


def _render_reliability(disease: str) -> None:
    st.subheader("Explanation reliability")
    report_path = RESULTS_DIR / disease / "reliability" / "reliability.json"
    if not report_path.exists():
        st.info("Reliability artifacts are not available yet.")
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    agreement = report["shap_permutation_agreement"]
    st.metric(
        "SHAP / permutation rank agreement",
        f"{agreement['spearman_rank_correlation']:.3f}",
        help="Spearman correlation between held-out mean absolute SHAP and ROC-AUC permutation importance rankings.",
    )
    stability = report["perturbation_experiment"]["stability_by_magnitude"]
    rows = [
        {
            "Perturbation (training SD)": values[
                "magnitude_in_training_standard_deviations"
            ],
            "Mean rank correlation": values["spearman_rank_correlation_mean"],
            "Rank correlation SD": values["spearman_rank_correlation_std"],
            "Mean top-5 Jaccard": values["top5_jaccard_mean"],
            "Top-5 Jaccard SD": values["top5_jaccard_std"],
        }
        for values in stability.values()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    plot_path = RESULTS_DIR / disease / "reliability" / "stability_by_magnitude.png"
    if plot_path.exists():
        st.image(str(plot_path), width="stretch")
    st.caption(report["interpretation_limitations"])


def _render_disease_page(disease: str) -> None:
    label = DISEASE_LABELS[disease]
    st.markdown(f'<div class="eyebrow">Disease module · {disease}</div>', unsafe_allow_html=True)
    st.title(label)
    try:
        bundle = load_disease_bundle(disease)
    except (FileNotFoundError, KeyError, ValueError, OSError) as error:
        st.error(f"Required project artifacts are unavailable: {error}")
        return

    prediction_tab, performance_tab, explanation_tab, reliability_tab = st.tabs(
        ["Prediction", "Performance", "Explainability", "Reliability"]
    )
    with prediction_tab:
        _render_prediction(disease, bundle)
    with performance_tab:
        _render_performance(disease, bundle)
    with explanation_tab:
        _render_explanations(disease, bundle)
    with reliability_tab:
        _render_reliability(disease)
    st.markdown(f'<div class="clinical-note">{DISCLAIMER}</div>', unsafe_allow_html=True)


def main() -> None:
    st.sidebar.markdown('<div class="eyebrow">Research workspace</div>', unsafe_allow_html=True)
    st.sidebar.title("Disease modules")
    page_names = ["Overview", *[DISEASE_LABELS[key] for key in FEATURE_SCHEMAS]]
    selected_page = st.sidebar.radio("Module", page_names, label_visibility="collapsed")
    st.sidebar.caption("Independent datasets, models, and explanations")

    if selected_page == "Overview":
        _render_home()
        return
    disease = next(key for key, label in DISEASE_LABELS.items() if label == selected_page)
    _render_disease_page(disease)


if __name__ == "__main__":
    main()
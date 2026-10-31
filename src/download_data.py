"""Download, validate, and save the three independent UCI disease datasets."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.data_validation import (
    DatasetValidationError,
    clean_missing_markers,
    map_binary_target,
    normalize_column_names,
    normalize_column_name,
    value_counts_as_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets"
HEART_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    uci_id: int
    filename: str
    expected_rows: int
    expected_features: tuple[str, ...]
    source_target: str
    target_mapping: dict[str, int]
    limitations: str


DATASET_SPECS = (
    DatasetSpec(
        key="diabetes",
        uci_id=529,
        filename="diabetes.csv",
        expected_rows=520,
        expected_features=(
            "age",
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
        ),
        source_target="class",
        target_mapping={"Positive": 1, "Negative": 0},
        limitations=(
            "Questionnaire records collected at Sylhet Diabetes Hospital in "
            "Bangladesh; not an external or prospective validation cohort."
        ),
    ),
    DatasetSpec(
        key="heart",
        uci_id=45,
        filename="heart.csv",
        expected_rows=303,
        expected_features=(
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ),
        source_target="num",
        target_mapping={"0": 0, "1": 1, "2": 1, "3": 1, "4": 1},
        limitations=(
            "This downloader accepts only the 303-row processed Cleveland table. "
            "The UCI resource also describes three other contributing cohorts."
        ),
    ),
    DatasetSpec(
        key="kidney",
        uci_id=336,
        filename="kidney.csv",
        expected_rows=400,
        expected_features=(
            "age",
            "bp",
            "sg",
            "al",
            "su",
            "rbc",
            "pc",
            "pcc",
            "ba",
            "bgr",
            "bu",
            "sc",
            "sod",
            "pot",
            "hemo",
            "pcv",
            "wbcc",
            "rbcc",
            "htn",
            "dm",
            "cad",
            "appet",
            "pe",
            "ane",
        ),
        source_target="class",
        target_mapping={"ckd": 1, "notckd": 0},
        limitations=(
            "A small, single-hospital dataset collected over approximately two "
            "months, as described by UCI; not a representative population sample."
        ),
    ),
)


class ClevelandProvenanceError(DatasetValidationError):
    """Raised unless the UCI Heart response is verifiably Cleveland-only."""


def _validate_metadata(spec: DatasetSpec, metadata: dict[str, Any]) -> None:
    if metadata.get("uci_id") != spec.uci_id:
        raise DatasetValidationError(
            f"Expected UCI id {spec.uci_id}; got {metadata.get('uci_id')!r}"
        )
    if metadata.get("num_instances") != spec.expected_rows:
        raise DatasetValidationError(
            f"UCI metadata for {spec.key} reports "
            f"{metadata.get('num_instances')!r} rows; expected {spec.expected_rows}."
        )
    if metadata.get("num_features") != len(spec.expected_features):
        raise DatasetValidationError(
            f"UCI metadata for {spec.key} reports an unexpected feature count."
        )

    if spec.key == "heart":
        additional_info = metadata.get("additional_info") or {}
        summary = additional_info.get("summary") or ""
        normalized_summary = re.sub(r"[^a-z0-9]+", " ", summary.casefold())
        is_cleveland_description = (
            "one file has been processed" in normalized_summary
            and "containing the cleveland database" in normalized_summary
        )
        if not is_cleveland_description:
            raise ClevelandProvenanceError(
                "UCI metadata no longer identifies its processed file as the "
                "Cleveland database. Refusing to download or save Heart data."
            )


def _build_dataset(spec: DatasetSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset = fetch_ucirepo(id=spec.uci_id)
    metadata = dataset.metadata
    _validate_metadata(spec, metadata)

    source_features = dataset.data.features
    source_targets = dataset.data.targets
    if len(source_features) != spec.expected_rows:
        error_type = ClevelandProvenanceError if spec.key == "heart" else DatasetValidationError
        raise error_type(
            f"Downloaded {spec.key} table has {len(source_features)} rows; "
            f"expected {spec.expected_rows}."
        )
    if source_targets.shape[1] != 1 or len(source_targets) != len(source_features):
        raise DatasetValidationError(
            f"{spec.key} returned an unexpected target table shape: "
            f"{source_targets.shape}."
        )

    normalized_features, name_mapping = normalize_column_names(source_features)
    if tuple(normalized_features.columns) != spec.expected_features:
        error_type = ClevelandProvenanceError if spec.key == "heart" else DatasetValidationError
        raise error_type(
            f"Unexpected {spec.key} feature schema. Expected "
            f"{spec.expected_features}, got {tuple(normalized_features.columns)}."
        )

    original_target_name = str(source_targets.columns[0])
    if normalize_column_name(original_target_name) != spec.source_target:
        raise DatasetValidationError(
            f"Expected target column {spec.source_target!r} for {spec.key}; "
            f"got {original_target_name!r}."
        )

    features = clean_missing_markers(normalized_features.reset_index(drop=True))
    raw_labels = clean_missing_markers(
        source_targets.iloc[:, 0].reset_index(drop=True).to_frame("source_target")
    )["source_target"]
    mapped_target = map_binary_target(raw_labels, spec.key)

    frame = features.copy()
    frame["target"] = mapped_target
    duplicate_count = int(frame.duplicated().sum())
    frame = frame.reset_index(drop=True)
    missing_target_count = int(frame["target"].isna().sum())
    clean_frame = frame.dropna(subset=["target"]).copy()
    clean_frame["target"] = clean_frame["target"].astype("int64")

    raw_distribution = value_counts_as_text(raw_labels)
    feature_missing_counts = {
        column: int(features[column].isna().sum()) for column in features.columns
    }
    clean_distribution = value_counts_as_text(clean_frame["target"])

    report = {
        "dataset": spec.key,
        "uci_id": spec.uci_id,
        "source_shape": [int(len(source_features)), int(source_features.shape[1] + 1)],
        "saved_shape": [int(clean_frame.shape[0]), int(clean_frame.shape[1])],
        "columns": list(clean_frame.columns),
        "feature_missing_value_counts": feature_missing_counts,
        "source_target_column": original_target_name,
        "target_labels_before_mapping": raw_distribution,
        "mapped_class_distribution": clean_distribution,
        "duplicate_rows_detected_for_preprocessing": duplicate_count,
        "missing_target_rows_removed": missing_target_count,
    }

    source_summary = (metadata.get("additional_info") or {}).get("summary") or ""
    manifest_entry = {
        "uci_id": spec.uci_id,
        "name": metadata.get("name"),
        "repository_url": metadata.get("repository_url"),
        "data_url": metadata.get("data_url"),
        "doi": metadata.get("dataset_doi"),
        "uci_last_updated": metadata.get("last_updated"),
        "source_rows": int(len(source_features)),
        "saved_rows": int(len(clean_frame)),
        "original_feature_names": [str(name) for name in source_features.columns],
        "normalized_feature_names": list(features.columns),
        "original_to_normalized_feature_names": name_mapping,
        "source_target_column": original_target_name,
        "target_column_in_csv": "target",
        "target_mapping": spec.target_mapping,
        "target_labels_before_mapping": raw_distribution,
        "mapped_class_distribution": clean_distribution,
        "feature_missing_value_counts": feature_missing_counts,
        "duplicate_rows_detected_for_preprocessing": duplicate_count,
        "missing_target_rows_removed": missing_target_count,
        "uci_source_summary": source_summary,
        "limitations": spec.limitations,
        "csv_file": spec.filename,
    }

    if spec.key == "heart":
        manifest_entry["cohort"] = "Cleveland"
        manifest_entry["cohort_verification"] = {
            "uci_metadata_identifies_processed_file_as_cleveland": True,
            "expected_and_observed_rows": spec.expected_rows,
            "expected_and_observed_feature_schema": list(spec.expected_features),
            "target_labels_verified_within": [0, 1, 2, 3, 4],
            "note": (
                "UCI metadata also describes multiple cohorts. Acquisition is "
                "refused unless the processed-Cleveland statement, 303-row "
                "count, and Cleveland feature schema all match."
            ),
        }

    return clean_frame, {"report": report, "manifest": manifest_entry}


def _write_outputs(
    output_dir: Path,
    datasets: dict[str, pd.DataFrame],
    entries: dict[str, dict[str, Any]],
    overwrite: bool,
) -> Path:
    manifest_path = output_dir / "manifest.json"
    expected_paths = [output_dir / spec.filename for spec in DATASET_SPECS]
    existing = [path for path in [*expected_paths, manifest_path] if path.exists()]
    if existing and not overwrite:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing data files: {formatted}. "
            "Pass --overwrite to replace them."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in DATASET_SPECS:
        datasets[spec.key].to_csv(output_dir / spec.filename, index=False)

    manifest = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "ucimlrepo_version": version("ucimlrepo"),
        "datasets": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _manual_heart_instructions() -> str:
    return (
        "\nHeart data was not saved. To load it manually, download the UCI "
        f"archive ({HEART_ARCHIVE_URL}), extract only "
        "processed.cleveland.data, and verify that it contains the Cleveland "
        "cohort's 303 records. Convert it to datasets/heart.csv with columns "
        f"{', '.join(DATASET_SPECS[1].expected_features)}, target, where "
        "target is 0 for num=0 and 1 for num in 1..4. Drop rows with missing "
        "num. Do not use new.data or concatenate the Hungarian, Switzerland, "
        "or VA files."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the validated CSV files and manifest.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated CSV and manifest files.",
    )
    args = parser.parse_args(argv)

    datasets: dict[str, pd.DataFrame] = {}
    entries: dict[str, dict[str, Any]] = {}
    try:
        for spec in DATASET_SPECS:
            try:
                frame, details = _build_dataset(spec)
            except DatasetValidationError as error:
                if spec.key == "heart" and not isinstance(
                    error, ClevelandProvenanceError
                ):
                    raise ClevelandProvenanceError(
                        f"Heart cohort verification failed: {error}"
                    ) from error
                raise
            datasets[spec.key] = frame
            entries[spec.key] = details["manifest"]
            print(json.dumps(details["report"], indent=2))

        manifest_path = _write_outputs(
            args.output_dir, datasets, entries, overwrite=args.overwrite
        )
    except ClevelandProvenanceError as error:
        print(f"ERROR: {error}{_manual_heart_instructions()}", file=sys.stderr)
        return 2
    except (DatasetValidationError, FileExistsError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Validated CSV files and manifest saved under: {args.output_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
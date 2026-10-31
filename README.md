# Explainable Multi-Disease AI

Academic research prototype for independent early-stage diabetes, Cleveland heart disease, and chronic kidney disease models with SHAP explanations and explanation-reliability experiments. It is not a clinically validated diagnostic tool.

## Environment

The project targets Python 3.12 and runs on CPU. In PowerShell from the project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the venv already exists, activate it or use its executable directly.

## Research workflow

```powershell
.\.venv\Scripts\python.exe -m src.download_data
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m src.compare_models
.\.venv\Scripts\python.exe -m src.evaluate
.\.venv\Scripts\python.exe -m src.explain
.\.venv\Scripts\python.exe -m src.reliability
```

Add `--overwrite` to a command to regenerate its saved artifacts. Dataset 45 ingestion checks UCI metadata, Cleveland feature schema, and the 303-record cohort size; it refuses to save if those checks fail.

## Each command handles one stage:
1. `download_data.py` fetches the three UCI datasets, checks labels and schemas, and writes CSVs plus a manifest. Heart data is accepted only when the Cleveland cohort checks pass.
2. `data_validation.py` provides reusable column-name normalization, missing-marker cleanup, and disease-specific target mapping. It preserves missing targets and rejects unexpected labels rather than silently assigning them to a class.
3. `preprocessing.py` cleans features and constructs pipelines. It keeps imputation, scaling, and encoding inside the model pipeline to avoid fitting preprocessing on validation or test data.
4. `train.py` makes a seeded, stratified 80/20 split per disease. It cross-validates Logistic Regression, Random Forest, and XGBoost using only the training portion, then saves the fitted models and split.
5. `compare_models.py` derives soft-voting weights from training-only out-of-fold predictions. The weights are nonnegative and sum to one; this step does not use the test split.
6. `evaluate.py` chooses thresholds from training-only predictions, then evaluates each model and ensemble on the reserved test split at both threshold 0.5 and the selected threshold. It saves metric tables and plots.
7. `explain.py` selects the individual model with the best mean training-CV ROC-AUC, then generates SHAP global and local explanations.
8. `reliability.py` checks how explanations change under numeric feature perturbations and compares SHAP rankings with permutation-importance rankings.

## Run the application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app reads the saved datasets, models, metrics, and explanation artifacts. Prediction inputs are not written to project files. Model scores are not validated clinical risk estimates.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Architecture

See [docs/system_architecture.md](docs/system_architecture.md) for the system diagram and module flowcharts.
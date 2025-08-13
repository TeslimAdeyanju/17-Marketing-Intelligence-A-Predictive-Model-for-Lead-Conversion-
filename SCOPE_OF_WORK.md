# Scope of Work — Marketing Intelligence: Predictive Model for Lead Conversion

## 1) Background

You have a Kaggle-derived Leads dataset (Leads.csv) and a working notebook framing a classification approach (baseline Logistic Regression) to predict whether a lead converts. The project aims to turn that exploration into a reproducible, documented lead scoring solution that marketing/sales can use.

## 2) Objectives and Success Criteria

Objectives

- Build an interpretable model to predict lead conversion probability.
- Identify key drivers of conversion for prioritization and messaging.
- Deliver a repeatable pipeline and a simple way to score new leads.

Success criteria (quantitative)

- Model AUC ≥ 0.80 on a held-out test set (or best-effort vs. baseline).
- Precision@Top-20% leads ≥ baseline heuristic by ≥ 10%.
- Calibration curve within acceptable deviation (Brier score tracked).

Success criteria (qualitative)

- Notebook runs top-to-bottom without manual edits.
- Clear explanations of top features and data limitations.
- Lightweight handoff docs and example scoring workflow.

## 3) In Scope

Data and understanding

- Load and audit `Leads.csv` (schema, types, duplicates, missingness, target distribution).
- Formalize missing value policy; drop high-null columns (>30–35%) when justified.
- Standardize column names (lowercase, underscores) per current notebook convention.

Feature prep and engineering

- Encode categoricals (One-Hot or Target encoding where appropriate and justified).
- Scale/normalize numeric features where needed for linear models.
- Handle class imbalance using `class_weight` and/or SMOTE (if added as dependency).

Modeling

- Baseline: Logistic Regression with regularization and class weights.
- Alternative(s): RandomForest or GradientBoosting (scikit-learn) for non-linear signal.
- Train/val/test split with fixed random seed; avoid leakage via proper pipelines.

Evaluation and interpretation

- Metrics: ROC AUC, PR AUC, Accuracy, Precision, Recall, F1, Confusion Matrix.
- Business metric: Precision@Top-K% (e.g., 10%, 20%).
- Feature importance: coefficients (LR) and impurity/SHAP-lite discussion for trees.
- Calibration check (reliability curve, Brier score).

Lead scoring and packaging

- Produce probability-based lead scores and bands:
  - High > 0.70, Medium 0.40–0.70, Low < 0.40 (tunable).
- Save trained model artifact with `joblib`.
- Provide `predict.py` CLI script to score a new CSV and emit `scores.csv`.

Documentation and reproducibility

- Finalize `notebook.ipynb` and ensure parity with `notebook.py`.
- Add concise `MODEL_CARD.md` (data, intended use, limitations, metrics).
- Update README with how-to-run and scoring example.

Optional (time-permitting)

- Simple Streamlit UI to upload a CSV and view scores/segments.
- Basic data validation (pydantic/schematics) for incoming scoring files.

## 4) Out of Scope

- Production deployment, CI/CD, or MLOps pipelines.
- Real-time inference or CRM integration (e.g., Salesforce, HubSpot).
- A/B testing or downstream campaign optimization.

## 5) Deliverables

- Cleaned and processed dataset snapshot (e.g., `data/processed/leads_clean.csv`).
- Reproducible training notebook and scripts under `src/` (pipeline + training + eval).
- Trained model artifact under `models/model.joblib` and `MODEL_CARD.md`.
- Evaluation report/figures under `reports/` (metrics, confusion matrix, calibration).
- Scoring CLI: `predict.py` that takes input CSV and emits scored output.
- Updated `README.md` with setup, train, evaluate, and score instructions.

## 6) Milestones and Timeline (indicative)

- M1 — Data audit and EDA (1–2 days)
  - Schema/type checks, missingness plan, target distribution, leakage checks.
- M2 — Baseline pipeline and metrics (2–3 days)
  - LR baseline, split strategy, evaluation, initial feature importance.
- M3 — Alternative model(s) and calibration (2 days)
  - RandomForest/GB baseline, compare, select, and calibrate if needed.
- M4 — Lead scoring + packaging (2 days)
  - Save model, implement CLI, generate example scores, create model card.
- M5 — Documentation and handoff (1 day)
  - README updates, final notebook polish, assumptions/limitations.
- Optional — Streamlit UI (1–2 days)

## 7) Assumptions and Constraints

- Data is provided as `Leads.csv` and is representative of current marketing operations.
- No PII exposure beyond what is already present in the dataset; OK to store locally.
- Python environment from `requirements.txt` (Python ≥ 3.10). If SMOTE is required, add `imbalanced-learn`.
- Stakeholder availability for quick feedback on business thresholds (Top-K%).

## 8) Risks and Mitigations

- High missingness / data quality: document drops/ imputations; sensitivity checks.
- Class imbalance: use class weights; if needed, add SMOTE with careful CV.
- Leakage: enforce pipeline-based transforms; time-aware split if temporal effects exist.
- Overfitting: cross-validation and hold-out test; prefer simpler models when similar performance.
- Drift: document monitoring suggestions; not implemented in this scope.

## 9) Acceptance Criteria

- All code and notebooks run end-to-end and reproduce reported metrics.
- Final chosen model meets or improves upon the baseline and hits at least one quantitative target (AUC or Precision@Top-K).
- `predict.py` successfully scores a sample CSV and outputs probabilities and bands.
- Documentation present: README usage, model card, and evaluation artifacts.

## 10) Proposed Folder Structure

```text
.
├── data/
│   ├── raw/Leads.csv
│   └── processed/leads_clean.csv
├── models/
│   └── model.joblib
├── reports/
│   ├── figures/
│   └── metrics.json
├── src/
│   ├── pipeline.py
│   ├── train.py
│   └── evaluate.py
├── predict.py
├── notebook.ipynb
├── MODEL_CARD.md
└── README.md
```

## 11) Notes on Current Repo

- Notebook already includes: column normalization helper, missingness policy (>30–35%), baseline LR imports.
- `requirements.txt` is present; `imbalanced-learn` not listed — add only if SMOTE is selected.
- The last attempt to run `marimo edit notebook.py` failed (exit 127). If you plan to use Marimo, ensure the package is installed and available in PATH.

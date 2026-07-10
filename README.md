# Bank Customer Churn Prediction Pipeline

An end-to-end ML pipeline that predicts which active bank customers are likely to churn, explains *why* on a per-customer basis using SHAP, and feeds a Power BI dashboard — all triggered by one command: `python run_pipeline.py`.

Built to practice production-style ML engineering, not just model training: SQL data validation, a full sklearn preprocessing pipeline, multi-model comparison with MLflow tracking, SHAP explainability, and automated reporting.

---

## Why this project

Most churn-prediction projects stop at "trained a model in a notebook, got 85% accuracy." I wanted to build the layer around the model that real teams actually need:

- Can I trust the input data? → SQL validation + rejection/duplicate tracking
- Which model is actually better, and how do I know? → MLflow-logged comparison
- Can I explain a single prediction to a non-technical stakeholder? → per-customer SHAP drivers
- Can someone else consume this without touching Python? → Power BI dashboard fed by structured CSVs

---

## Architecture

```
Raw CSV → MySQL (staging)
        → sp_clean_and_validate_data()   [reject / dedupe / clean]
        → SQL views (feature engineering + data-quality report)
        → sklearn preprocessing (impute, scale, one-hot, SMOTE)
        → 3-model training loop (MLflow-logged) → best model by F1
        → predictions on active customers
        → SHAP explanations (per-customer + global)
        → Power BI (3-page dashboard)
```

---

## Results

| Model | Validation F1 |
|---|---|
| Logistic Regression | 0.507 |
| **Random Forest (selected)** | **0.600** |
| XGBoost | 0.588 |

- 10,150 raw records → 498 rejected (4.9%), 145 duplicates removed (1.4%) → 9,507 clean records
- 7,559 active customers scored
- F1 was the selection metric, not accuracy, since churn is class-imbalanced and a "predict everyone stays" model would score ~80% accuracy while catching zero churners

**Honest take on 0.60 F1:** this is a reasonable baseline, not a strong result. The gap is mostly a signal problem — the strongest available features (age, product count, activity status) are genuinely noisy predictors of churn. Next steps to push this further: hyperparameter search (currently untuned defaults), threshold tuning instead of the default 0.5 cutoff, and richer behavioral features (transaction frequency, recency) if the source data allows it.

**Top churn drivers (SHAP, global importance):**
1. `age` — mid-30s customers most retained; 44–51 is the highest-risk band
2. `products_number` — customers with 2 products are strongly retained; 1-product customers are highest risk
3. `gender` — smaller effect, slightly higher churn tendency for female customers

---

## What's in this repo

```
01_data_quality_setup.sql       SQL: validation, dedup, staging tables + stored procedure
02_analytical_views.sql         SQL: feature-engineering and data-quality views
import_csv.py                   One-time CSV → MySQL loader
run_pipeline.py                 Orchestrates the full pipeline end to end
src/
  db_connector.py                MySQL access layer
  preprocessing.py                Feature pipeline: impute, scale, OHE, SMOTE
  train.py                        3-model training loop, MLflow logging, SHAP
  predict.py                      Scores active customers
generate_feature_impact_csvs.py  Auto-generates Power BI-ready impact tables for the top 3 SHAP features
```

Outputs (`data_quality_metrics.csv`, `prediction_results.csv`, `dynamic_insights.csv`, `global_feature_importance.csv`, `feature_impact_csvs/`) feed a 3-page Power BI dashboard that auto-updates its titles/visuals based on whichever features the current model run ranks highest — the dashboard doesn't need manual edits when the pipeline is re-run on new data.

---

## Technical notes worth knowing

- **Data validation lives in SQL, not Python.** A stored procedure rejects bad records (missing IDs, unparseable birth dates, implausible ages) into a dedicated table with a reason code, and deduplicates via `ROW_NUMBER() OVER (PARTITION BY customer_id)`. This was a deliberate choice to keep data-quality logic close to the data.
- **Smart binning for Power BI:** numerical features get binned differently depending on their range (single bin, percentile-spaced, exact-integer, or 6–8 smart bins with outlier clamping at the P2–P98 range) so charts stay readable regardless of the feature's distribution — this logic is the most involved part of the codebase.
- **Known limitation — "active customers":** the dataset had no unlabeled (`NULL` churn) rows in the run this README documents, so predictions fall back to scoring customers labeled `churn = 0`. That's useful for a "who's at risk among people we currently think are safe" framing, but it's not the same as scoring genuinely unseen data — worth being upfront about if asked.
- **`age` is computed live** from date of birth at query time (`TIMESTAMPDIFF(YEAR, dob, CURDATE())`), so feature values (and therefore importances) will drift slightly between runs on different dates.
- Single 80/20 stratified split, no cross-validation — reasonable for a project of this scope, but the F1 numbers should be read as a single estimate, not a stable average.

---

## Stack

Python (pandas, scikit-learn, imbalanced-learn, XGBoost, SHAP, MLflow) · MySQL (stored procedures, window functions, views) · Power BI · SQLAlchemy/PyMySQL

---

## Running it

```bash
pip install -r requirements.txt
mysql -u root -p creditcard < 01_data_quality_setup.sql
mysql -u root -p creditcard < 02_analytical_views.sql
python import_csv.py        # one-time load
python run_pipeline.py      # full pipeline, every run
```

> DB credentials are read from environment variables (`DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`) — see `.env.example`.
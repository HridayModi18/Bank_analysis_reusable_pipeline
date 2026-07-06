# 🏦 Bank Customer Churn Prediction Pipeline

> **A production-grade, single-click end-to-end ML pipeline** combining SQL data engineering, Python machine learning, SHAP explainability, MLflow experiment tracking, and Power BI dashboarding — all orchestrated in a single command.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Business Problem](#2-business-problem)
3. [Architecture & Data Flow](#3-architecture--data-flow)
4. [Dataset Schema](#4-dataset-schema)
5. [Project Structure](#5-project-structure)
6. [Technology Stack](#6-technology-stack)
7. [Database Layer — SQL Scripts](#7-database-layer--sql-scripts)
   - [01_data_quality_setup.sql](#71-01_data_quality_setupsql)
   - [02_analytical_views.sql](#72-02_analytical_viewssql)
8. [Python Pipeline — Source Modules](#8-python-pipeline--source-modules)
   - [import_csv.py](#81-import_csvpy)
   - [db_connector.py](#82-db_connectorpy)
   - [preprocessing.py](#83-preprocessingpy)
   - [train.py](#84-trainpy)
   - [predict.py](#85-predictpy)
   - [generate_feature_impact_csvs.py](#86-generate_feature_impact_csvspy)
   - [run_pipeline.py](#87-run_pipelinepy)
9. [Pipeline Execution — Step by Step](#9-pipeline-execution--step-by-step)
10. [ML Models — Training & Selection](#10-ml-models--training--selection)
11. [SHAP Explainability Engine](#11-shap-explainability-engine)
12. [Smart Binning Algorithm](#12-smart-binning-algorithm)
13. [Output Files & Artifacts](#13-output-files--artifacts)
14. [Actual Execution Results](#14-actual-execution-results)
15. [MLflow Experiment Tracking](#15-mlflow-experiment-tracking)
16. [Power BI Dashboard Architecture](#16-power-bi-dashboard-architecture)
17. [Feature Importance Results](#17-feature-importance-results)
18. [How to Run](#18-how-to-run)
19. [Dependencies & Installation](#19-dependencies--installation)
20. [Configuration Reference](#20-configuration-reference)

---

## 1. Project Overview

This project is a **fully automated Bank Customer Churn Prediction System** built as an end-to-end data science pipeline. It was designed to demonstrate production-level skills across:

- **Data Engineering** — SQL stored procedures, validation pipelines, deduplication using window functions
- **Machine Learning** — multi-model training loop with automatic best-model selection
- **Explainability** — SHAP values for per-customer and global feature attribution
- **Experiment Tracking** — MLflow logging of all metrics, parameters, and model artifacts
- **Visualization** — structured CSV exports designed to feed a live 3-page Power BI dashboard

The pipeline runs entirely from one command: `python run_pipeline.py`

---

## 2. Business Problem

Banks face significant revenue loss when customers close their accounts. The goal of this project is:

1. **Identify** which *currently active* customers are most likely to churn (leave the bank).
2. **Explain** the primary factors driving each individual customer's churn risk.
3. **Monitor** the quality and health of incoming data.
4. **Visualize** findings in a business-friendly Power BI dashboard.

The dataset contains customer records in a MySQL database. Customers with a known `churn` label (0 or 1) are used for **model training**. Customers where `churn` is `NULL` (or `0` in the fallback case) are treated as **active customers** whose risk is predicted.

---

## 3. Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        END-TO-END PIPELINE FLOW                             │
└─────────────────────────────────────────────────────────────────────────────┘

 [RAW CSV FILE]
      │
      │  import_csv.py  (one-time bulk load)
      ▼
 [MySQL: bankChurn table]  ←── stg_staging / raw data
      │
      │  run_pipeline.py → db_connector.run_data_quality_procedure()
      ▼
 [sp_clean_and_validate_data()]  ←── Stored Procedure
      ├──► reject_records      (NULL customer_id, bad ages, unparseable dates)
      ├──► duplicate_records   (duplicate customer_id via ROW_NUMBER window fn)
      └──► bankchurn_clean     (validated, deduplicated records)
      │
      │  02_analytical_views.sql
      ▼
 [vw_Analytics_Features]         ←── Feature-enriched SQL View
 [vw_Data_Quality_Report]        ←── Health metrics SQL View
      │
      │  db_connector.fetch_analytics_features()
      │  db_connector.fetch_data_quality_metrics()
      ▼
 [Python: preprocessing.py]
      ├── Filter: labeled rows → Training set
      ├── Drop: identifiers (nome, email, numero_conta, data_nascimento)
      ├── Split: 80/20 stratified train/validation
      ├── ColumnTransformer:
      │     ├── Numerical: Median Imputer → StandardScaler
      │     └── Categorical: OneHotEncoder (handle_unknown='ignore')
      └── SMOTE: Over-sample minority class
      │
      ▼
 [Python: train.py]
      ├── Logistic Regression   → MLflow run → F1: 0.5069
      ├── Random Forest         → MLflow run → F1: 0.5997  ← WINNER
      └── XGBoost               → MLflow run → F1: 0.5884
      │
      ▼
 [Python: predict.py]
      └── Active customers → churn_probability, churn_predicted_class
      │                    → prediction_results.csv
      ▼
 [Python: train.generate_shap_insights()]
      ├── SHAP TreeExplainer (Random Forest)
      ├── Per-customer: top_1 driver, top_2 driver, SHAP impacts
      │                → dynamic_insights.csv
      └── Global: mean(|SHAP|) aggregated per original feature
               → global_feature_importance.csv
               → raw_global_feature_importance.csv
      │
      ▼
 [Python: generate_feature_impact_csvs.py]
      └── Top 3 features → per-feature retention/churn impact CSVs
                        → feature_impact_csvs/age_impact.csv
                        → feature_impact_csvs/products_number_impact.csv
                        → feature_impact_csvs/gender_impact.csv
      │
      ▼
 [Power BI Dashboard]
      ├── Page 1: Data Quality Health   ← data_quality_metrics.csv
      ├── Page 2: Business Analytics    ← prediction_results.csv
      └── Page 3: ML Insights (Dynamic) ← dynamic_insights.csv
                                          feature_impact_csvs/
```

---

## 4. Dataset Schema

The raw data is loaded into the `bankChurn` MySQL table (via `import_csv.py`). Each row represents one bank customer.

| Column | Type | Description |
|---|---|---|
| `customer_id` | `BIGINT` / PK | Unique customer identifier |
| `nome` | `VARCHAR(255)` | Customer full name |
| `email` | `VARCHAR(255)` | Customer email address |
| `numero_conta` | `VARCHAR(50)` | Bank account number |
| `data_nascimento` | `VARCHAR(50)` | Date of birth (accepts `DD/MM/YYYY` or `YYYY-MM-DD`) |
| `cidade` | `VARCHAR(100)` | City of residence |
| `country` | `VARCHAR(100)` | Country of residence |
| `gender` | `VARCHAR(20)` | Gender (Male / Female) |
| `age` | `INT` | Age in years (staging column; recalculated from DOB in view) |
| `credit_score` | `INT` | Credit score (300–850 typical range) |
| `tenure` | `INT` | Months as a bank customer |
| `balance` | `DOUBLE` | Current account balance |
| `limite_credito` | `DOUBLE` | Credit card limit |
| `products_number` | `INT` | Number of bank products held (1–4) |
| `credit_card` | `INT` | Has credit card (1 = yes, 0 = no) |
| `active_member` | `INT` | Is active member (1 = yes, 0 = no) |
| `estimated_salary` | `DOUBLE` | Estimated salary (can be NULL) |
| `churn` | `INT` | Target label (1 = churned, 0 = retained, NULL = active/unknown) |

**Engineered columns** added by the `vw_Analytics_Features` SQL view:

| Column | Type | Description |
|---|---|---|
| `calculated_age` | `INT` | Age recomputed from `data_nascimento` using `TIMESTAMPDIFF(YEAR, ...)` |
| `age_band` | `VARCHAR` | Age group bucket: `18-30`, `31-45`, `46-60`, `60+` |
| `credit_utilization` | `DOUBLE` | `balance / NULLIF(limite_credito, 0)` — avoids division by zero |
| `salary_missing_flag` | `INT` | `1` if `estimated_salary` is NULL, else `0` |

---

## 5. Project Structure

```
BankProject/
│
├── 📄 README.md                          ← This file
├── 📄 implementation_plan.md             ← Original technical design document
│
├── ── SQL Database Layer ──
├── 📄 01_data_quality_setup.sql          ← Creates tables + stored procedure
├── 📄 02_analytical_views.sql            ← Creates feature + quality views
│
├── ── Python Pipeline Root ──
├── 📄 import_csv.py                      ← One-time CSV → MySQL loader
├── 📄 run_pipeline.py                    ← Main orchestration script (ENTRY POINT)
├── 📄 generate_feature_impact_csvs.py    ← SHAP-powered feature impact generator
│
├── ── Python Source Modules ──
├── src/
│   ├── 📄 db_connector.py               ← MySQL connection + query functions
│   ├── 📄 preprocessing.py              ← Feature engineering + SMOTE
│   ├── 📄 train.py                      ← Model training, selection, SHAP
│   └── 📄 predict.py                    ← Active customer prediction
│
├── ── Raw Data ──
├── 📄 bank_churn_ascii.csv              ← Original clean dataset (ASCII encoded)
├── 📄 bank_churn_dirty.csv             ← Dirty dataset with validation issues
│
├── ── Pipeline Outputs ──
├── 📄 data_quality_metrics.csv          ← Data health report
├── 📄 prediction_results.csv            ← Churn probabilities for active customers
├── 📄 dynamic_insights.csv              ← Per-customer SHAP top drivers
├── 📄 global_feature_importance.csv     ← Aggregated global SHAP importances
├── 📄 raw_global_feature_importance.csv ← Per-dummy-column SHAP importances
│
├── ── Feature Impact CSVs ──
├── feature_impact_csvs/
│   ├── 📄 age_impact.csv               ← Age bin vs retention/churn impact
│   ├── 📄 gender_impact.csv            ← Gender vs retention/churn impact
│   └── 📄 products_number_impact.csv   ← Products count vs retention/churn impact
│
├── ── Experiment Tracking ──
├── mlruns/                              ← MLflow run artifacts (auto-generated)
├── mlflow.db                            ← MLflow SQLite backend
│
├── ── Logs ──
├── 📄 pipeline_execution_log.txt        ← Captured stdout/stderr from last run
├── 📄 pipeline_output.log               ← Short execution summary
│
└── venv/                                ← Python virtual environment
```

---

## 6. Technology Stack

| Category | Technology | Purpose |
|---|---|---|
| **Database** | MySQL 8.0+ | Stores raw, clean, reject, and duplicate records |
| **SQL** | Standard SQL + Window Functions | Data validation, deduplication, feature views |
| **Python** | Python 3.10+ | ML pipeline, orchestration, data processing |
| **ORM / DB Driver** | `pymysql` + `SQLAlchemy` | MySQL connectivity from Python |
| **Data Manipulation** | `pandas`, `numpy` | DataFrames, numerical operations |
| **ML Framework** | `scikit-learn` | Preprocessing pipelines, LR, RF, metrics |
| **Boosting** | `xgboost` | XGBoost classifier |
| **Class Balancing** | `imbalanced-learn` (SMOTE) | Synthetic Minority Over-sampling |
| **Explainability** | `shap` | SHAP values for feature attribution |
| **Experiment Tracking** | `mlflow` | Logging runs, metrics, parameters, models |
| **Visualization** | Power BI | Interactive 3-page business dashboard |

---

## 7. Database Layer — SQL Scripts

### 7.1 `01_data_quality_setup.sql`

**Purpose:** Sets up the entire SQL data quality infrastructure from scratch. This script is run once (during database setup) before any pipeline execution.

#### Tables Created

**`bankchurn_clean`** — The single source of truth.
```sql
CREATE TABLE bankchurn_clean (
    customer_id  BIGINT PRIMARY KEY,
    nome         VARCHAR(255),
    email        VARCHAR(255),
    numero_conta VARCHAR(50),
    data_nascimento VARCHAR(50),
    cidade       VARCHAR(100),
    country      VARCHAR(100),
    gender       VARCHAR(20),
    age          INT,
    credit_score INT,
    tenure       INT,
    balance      DOUBLE,
    limite_credito DOUBLE,
    products_number INT,
    credit_card  INT,
    active_member INT,
    estimated_salary DOUBLE,
    churn        INT
);
```

**`reject_records`** — Holds rows that fail validation. Includes an extra `rejection_reason VARCHAR(255)` column to track *why* each record was rejected.

**`duplicate_records`** — Holds duplicate rows identified by `customer_id`. Same schema as `reject_records` minus the reason column.

#### Stored Procedure: `sp_clean_and_validate_data()`

This is the core of the SQL data quality layer. It runs atomically and performs three operations:

**Step 1 — Rejection Logic**

Records are rejected into `reject_records` if any of these conditions are met:

| Condition | Rejection Reason |
|---|---|
| `customer_id IS NULL` | `'Missing Customer ID'` |
| `data_nascimento IS NULL` | `'Missing Birth Date'` |
| Date cannot be parsed in either `DD/MM/YYYY` or `YYYY-MM-DD` format | `'Unparseable Birth Date Format'` |
| Computed age < 18 | `'Age below 18'` |
| Computed age > 110 | `'Age above 110'` |
| `age = -1` | `'Staging Age is -1'` |

The date parsing uses a dual-format `CASE` statement:
```sql
CASE 
    WHEN data_nascimento LIKE '%/%/%' THEN STR_TO_DATE(data_nascimento, '%d/%m/%Y') 
    WHEN data_nascimento LIKE '%-%-%' THEN STR_TO_DATE(data_nascimento, '%Y-%m-%d') 
    ELSE NULL 
END
```

Age validation uses `TIMESTAMPDIFF(YEAR, parsed_date, CURDATE())` to compute age dynamically at procedure call time.

**Step 2 — Deduplication via Window Functions**

A temporary table `temp_valid_partitioned` is created using `ROW_NUMBER()` to rank duplicate `customer_id` entries:

```sql
ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY (SELECT NULL)) as row_num
```

- Rows where `row_num > 1` → inserted into `duplicate_records`
- Rows where `row_num = 1` → inserted into `bankchurn_clean`

**Step 3 — Type-safe Insert**

`customer_id` is explicitly cast to `SIGNED` integer on insert into `bankchurn_clean` to prevent type mismatches from raw staging data.

---

### 7.2 `02_analytical_views.sql`

**Purpose:** Creates two SQL views that act as the semantic layer — transforming raw cleaned data into model-ready features and health metrics.

#### View 1: `vw_Analytics_Features`

This view uses a **CTE chain** (Common Table Expressions) to progressively compute derived columns:

```sql
WITH CalculatedDOB AS (
    SELECT *, 
        CASE 
            WHEN data_nascimento LIKE '%/%/%' THEN STR_TO_DATE(data_nascimento, '%d/%m/%Y') 
            WHEN data_nascimento LIKE '%-%-%' THEN STR_TO_DATE(data_nascimento, '%Y-%m-%d') 
        END AS parsed_dob
    FROM bankchurn_clean
),
CalculatedAge AS (
    SELECT *, 
        TIMESTAMPDIFF(YEAR, parsed_dob, CURDATE()) AS calculated_age
    FROM CalculatedDOB
)
SELECT ...
```

**Features computed by this view:**

| Feature | SQL Logic |
|---|---|
| `age` | `TIMESTAMPDIFF(YEAR, parsed_dob, CURDATE())` — live age computation |
| `age_band` | CASE bucketing: `18-30`, `31-45`, `46-60`, `60+` |
| `cidade` | `LOWER(TRIM(cidade))` — normalized city name |
| `country` | `LOWER(TRIM(country))` — normalized country name |
| `credit_utilization` | `balance / NULLIF(limite_credito, 0)` |
| `salary_missing_flag` | `CASE WHEN estimated_salary IS NULL THEN 1 ELSE 0 END` |

#### View 2: `vw_Data_Quality_Report`

A single-row aggregate view providing data health metrics:

| Column | Description |
|---|---|
| `total_loaded` | Total raw records in `bankChurn` |
| `total_rejected` | Records in `reject_records` |
| `total_duplicates` | Records in `duplicate_records` |
| `total_clean` | Records in `bankchurn_clean` |
| `salary_missing_pct` | Percentage of records with NULL `estimated_salary` |

---

## 8. Python Pipeline — Source Modules

### 8.1 `import_csv.py`

**Purpose:** One-time utility script to bulk-load the raw CSV file into the MySQL `bankchurn` table.

```python
df = pd.read_csv("/path/to/bank_churn_ascii.csv")
engine = create_engine("mysql+pymysql://root:password@localhost:3306/creditcard")
df.to_sql("bankchurn", con=engine, if_exists="replace", index=False)
```

- Uses `pandas.DataFrame.to_sql()` with `if_exists="replace"` — **drops and recreates** the table each time.
- Loads into the `creditcard` MySQL database.
- This script is run **once manually** before the pipeline.

---

### 8.2 `src/db_connector.py`

**Purpose:** Encapsulates all database connectivity. Acts as the data access layer for the pipeline.

**Database config (hardcoded):**

| Parameter | Value |
|---|---|
| `DB_HOST` | `localhost` |
| `DB_PORT` | `3306` |
| `DB_USER` | `root` |
| `DB_NAME` | `creditcard` |

**Functions:**

#### `get_connection()`
Creates a raw `pymysql` connection. Used when executing stored procedures (requires cursor-level control).

#### `get_sqlalchemy_engine()`
Creates a `SQLAlchemy` engine using the `mysql+pymysql://` dialect. Required for `pd.read_sql()` compatibility.

```python
connection_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
return create_engine(connection_url)
```

#### `run_data_quality_procedure()`
Executes the stored procedure:
```python
cursor.execute("CALL sp_clean_and_validate_data()")
connection.commit()
```
Wraps in `try/except/finally` to guarantee connection close.

#### `fetch_analytics_features()`
Fetches all rows from `vw_Analytics_Features`:
```python
df = pd.read_sql("SELECT * FROM vw_Analytics_Features", con=engine)
```
Returns a pandas DataFrame. In the actual run, this returned **9,507 records**.

#### `fetch_data_quality_metrics()`
Fetches the single-row report from `vw_Data_Quality_Report`:
```python
df = pd.read_sql("SELECT * FROM vw_Data_Quality_Report", con=engine)
```

---

### 8.3 `src/preprocessing.py`

**Purpose:** Full feature engineering and preprocessing pipeline for model training and prediction.

#### Feature Column Configuration

```python
categorical_features = ['gender', 'cidade', 'country']
numerical_features = [
    'credit_score', 'age', 'tenure', 'balance', 'limite_credito', 
    'products_number', 'credit_card', 'active_member', 'estimated_salary',
    'credit_utilization', 'salary_missing_flag'
]
```

**Note:** `age_band` is NOT included as a raw ML feature (it is used as a SHAP explanation dimension only). This prevents encoding redundancy with `age`.

#### `build_preprocessing_pipeline()`

Returns a `ColumnTransformer` with two sub-pipelines:

**Numerical Pipeline:**
```python
Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Handles NULL estimated_salary
    ('scaler', StandardScaler())                     # Z-score normalization
])
```

**Categorical Pipeline:**
```python
Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
```
- `handle_unknown='ignore'` ensures unseen categories during prediction don't crash the pipeline.
- `sparse_output=False` returns a dense array for compatibility.

#### `prepare_training_data(df)`

Full training data preparation:

1. **Filter:** `df[df['churn'].notna()]` — only rows with known labels
2. **Drop PII:** removes `customer_id`, `nome`, `email`, `numero_conta`, `data_nascimento`
3. **Target:** `y = train_df['churn'].astype(int)`
4. **Split:** `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`
5. **Fit preprocessor** on training set, **transform** both train and val
6. **Extract OHE feature names** using `get_feature_names_out()` for SHAP compatibility
7. **SMOTE:** `SMOTE(random_state=42).fit_resample(X_train_processed, y_train)`

SMOTE results in this run:
- Original training shape: **(7,605, 39)**
- Resampled (after SMOTE) shape: **(12,094, 39)**

**Returns a dictionary:**
```python
{
    'X_train': X_train_resampled,   # SMOTE-balanced training features
    'y_train': y_train_resampled,   # SMOTE-balanced training labels
    'X_val': X_val_processed,       # Unmodified validation features
    'y_val': y_val,                 # Unmodified validation labels
    'preprocessor': preprocessor,   # Fitted ColumnTransformer (for prediction)
    'feature_names': final_feature_names  # 39 feature names post-OHE
}
```

#### `prepare_prediction_data(df, preprocessor)`

Prepares active customers for prediction:

1. Selects rows where `churn IS NULL` (preferred) — these are customers with unknown status.
2. Falls back to `churn == 0` if no NULL rows exist (as occurred in the actual run).
3. Drops the same PII columns as training.
4. Applies the **already-fitted** preprocessor using `.transform()` only (no re-fitting).
5. Returns `(active_customer_ids, X_active_processed)`.

---

### 8.4 `src/train.py`

**Purpose:** Multi-model training loop, MLflow logging, SHAP explainability, and global feature importance export.

#### `train_and_evaluate(X_train, y_train, X_val, y_val, feature_names)`

Trains three candidate models and selects the best by **F1-Score on the validation set**.

**Candidate models:**

| Model | Configuration |
|---|---|
| `Logistic_Regression` | `max_iter=1000, random_state=42` |
| `Random_Forest` | `n_estimators=100, max_depth=12, random_state=42` |
| `XGBoost` | `n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='logloss', random_state=42` |

**For each model, MLflow logs:**
- All model hyperparameters via `mlflow.log_params(model.get_params())`
- `validation_f1_score`
- `validation_precision`
- `validation_recall`
- `validation_accuracy`
- Model artifact (XGBoost uses `mlflow.xgboost.log_model`, others use `mlflow.sklearn.log_model`)

**Selection criterion:**
```python
if f1 > best_f1:
    best_f1 = f1
    best_model_name = model_name
    best_model_obj = model
```

Returns `(best_model_obj, best_model_name)`.

#### `get_original_value(feature_name, customer_id, active_df, scaled_val)`

A helper to **reverse-look up original (unscaled) feature values** for SHAP output, so the exported insights show meaningful values (e.g., `age = 45`, not `age = 0.73`).

- For direct columns (e.g., `age`): fetches `active_df.loc[customer_id, feature_name]`
- For OHE columns (e.g., `gender_Male`): strips prefix, matches raw value, returns `1.0` or `0.0`
- Falls back to the raw scaled value (rounded to 4 decimals) on any exception.

#### `generate_shap_insights(best_model, X_active_processed, active_df, active_customer_ids, feature_names)`

**Full SHAP pipeline for explainability:**

1. **Explainer selection:**
   - `LogisticRegression` → `shap.LinearExplainer`
   - All tree models (RF, XGBoost) → `shap.TreeExplainer`
   - Fallback: `shap.Explainer` (model-agnostic)

2. **Shape handling:** 3D SHAP output from `TreeExplainer` on binary classifiers is sliced to `shap_values[:, :, 1]` (class 1 = churn).

3. **Per-customer top drivers:**
   ```python
   sorted_indices = np.argsort(customer_shap)[::-1]
   top_1_idx = sorted_indices[-1]   # Most negative SHAP (retention driver)
   top_2_idx = sorted_indices[0]    # Most positive SHAP (churn driver)
   ```
   - `top_1` = the feature *most strongly pushing toward retention* (negative SHAP)
   - `top_2` = the feature *most strongly pushing toward churn* (positive SHAP)

4. **Exports `dynamic_insights.csv`** with columns:
   `customer_id`, `churn_probability`, `top_1_name`, `top_1_value`, `top_1_impact`, `top_2_name`, `top_2_value`, `top_2_impact`

5. **Global importance — raw:** `mean(|SHAP|)` across all active customers, per post-OHE feature column → `raw_global_feature_importance.csv`

6. **Global importance — aggregated:** OHE dummies are summed back to their original feature group (e.g., `gender_Male` + `gender_Female` → `gender`) → `global_feature_importance.csv`

---

### 8.5 `src/predict.py`

**Purpose:** Runs prediction on active customers and exports the results.

#### `generate_predictions(best_model, X_active_processed, active_customer_ids, original_full_df)`

1. Gets **probability scores**: `model.predict_proba(X_active_processed)[:, 1]` (column 1 = churn probability)
2. Gets **class predictions**: `model.predict(X_active_processed)` (binary 0 or 1)
3. Fetches the original rows from the full DataFrame for the active customers
4. Attaches `churn_probability` (rounded to 4 decimals) and `churn_predicted_class` to each row
5. Exports to `prediction_results.csv` — **all original columns are preserved**, plus the two new prediction columns

In the actual run, **7,559 active customers** were predicted.

---

### 8.6 `generate_feature_impact_csvs.py`

**Purpose:** Reads the top 3 features from `global_feature_importance.csv` and generates one Power BI-ready CSV per feature, showing how each feature value/bin impacts churn vs. retention.

This is the most algorithmically sophisticated module in the project. It contains a full **smart binning system**.

#### Entry Point: `generate_feature_impact_csvs()`

```python
global_df    = pd.read_csv("global_feature_importance.csv")
top_features = global_df.head(3)["feature_name"].tolist()
# → ['age', 'products_number', 'gender']
```

For each top feature, it detects whether it's **categorical** (OHE-based) or **numerical**, then calls the appropriate builder.

#### Feature Type Detection

```python
OHE_BASE_NAMES = ["country", "cidade", "gender", "age_band"]

def is_categorical(feature_name):
    return feature_name in OHE_BASE_NAMES
```

#### Smart Binning Algorithm (Numerical Features)

The `compute_smart_bins()` function applies a **5-rule binning decision tree**:

| Condition | Bins | Method |
|---|---|---|
| `range == 0` (all same) | 1 bin | Single label around the value |
| `range < 1` (fractional) | Up to 3 bins | P33 / P67 percentile-spaced |
| `1 ≤ range ≤ 10` | No bins | Each distinct integer = its own row |
| `10 < range ≤ 24` | 6 bins | Smart integer-aligned edges |
| `range > 24` | 8 bins | Smart integer-aligned edges |

**Outlier-aware edge placement:** For ranges > 10, interior bin edges are computed within the **P2–P98 effective range** rather than raw [min, max]. Outlier clamping is only activated if the outlier tail exceeds **20% of the P2–P98 core spread**. This prevents a handful of extreme values from creating useless wide bins.

**Integer alignment:** All interior bin edges are rounded to the nearest integer, ensuring bin boundaries like `"25–32"` instead of `"24.7–32.3"`.

#### `build_categorical_csv(feature_name, insights_df)`

For OHE features (like `gender`):
- **Retention side:** rows where `top_1_name` starts with `gender_` AND `top_1_value == 1.0`
- **Churn side:** rows where `top_2_name` starts with `gender_` AND `top_2_value == 1.0`
- Groups by category, aggregates `sum(impact)` and `count()` per category
- Merges both sides on category label with `outer` join (fills missing with 0)

**Output columns:** `value, retention_impact, num_retention_customers, churn_impact, num_churn_customers`

#### `build_exact_value_csv(feature_name, insights_df, true_vals)`

For small-range numerical features (1 ≤ range ≤ 10), like `products_number`:
- Rounds values to nearest integer so `2.0` and `2` group together
- Fills in the complete integer range `[min..max]` with zeros for missing values
- Produces one row per distinct integer value

#### `build_numerical_csv(feature_name, insights_df, true_vals)`

For wide-range numerical features (like `age`):
- Applies `compute_smart_bins()` to get edges and labels
- Uses `pd.cut(right=False, include_lowest=True)` for bin assignment
- Ensures all bin labels appear in output (fills empty bins with 0)
- Output column: `value_bin` (not `value`)

---

### 8.7 `run_pipeline.py`

**Purpose:** The single-click master orchestration script. Imports and sequences all modules.

```python
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
```

Adds the `src/` directory to `sys.path` so all source modules are importable without package installation.

**Execution sequence:**

| Step | Function Called | Output |
|---|---|---|
| 1 | `db_connector.run_data_quality_procedure()` | Cleans DB tables |
| 2 | `db_connector.fetch_data_quality_metrics()` | `data_quality_metrics.csv` |
| 3 | `db_connector.fetch_analytics_features()` | `features_raw_df` DataFrame |
| 4 | `preprocessing.prepare_training_data()` | `training_data` dict |
| 5 | `train.train_and_evaluate()` | `best_model`, `best_model_name` |
| 6 | `preprocessing.prepare_prediction_data()` | `active_ids`, `X_active_processed` |
| 7 | `predict.generate_predictions()` | `prediction_results.csv` |
| 8 | `train.generate_shap_insights()` | `dynamic_insights.csv`, `global_feature_importance.csv` |
| 9 | `generate_feature_impact_csvs.generate_feature_impact_csvs()` | `feature_impact_csvs/*.csv` |

Steps 6–9 are **skipped** (with a message) if no active customer records are found.

---

## 9. Pipeline Execution — Step by Step

Here is exactly what happens when you run `python run_pipeline.py`:

```
==================================================
Starting Bank Customer Churn Pipeline Execution
==================================================

--- Step 1: Executing database validation & cleaning procedure ---
→ Calls sp_clean_and_validate_data() via pymysql
→ Clears and repopulates: bankchurn_clean, reject_records, duplicate_records

--- Step 2: Fetching data quality metrics ---
→ Reads vw_Data_Quality_Report
→ Saves data_quality_metrics.csv

--- Step 3: Fetching analytics features from DB view ---
→ Reads vw_Analytics_Features
→ Returns DataFrame with engineered features

--- Step 4: Preprocessing and balancing class distribution ---
→ Filters labeled rows, drops PII columns
→ Splits 80/20 stratified
→ Fits ColumnTransformer (StandardScaler + OHE)
→ Runs SMOTE to balance minority class

--- Step 5: Training model loop & selection ---
→ Trains Logistic Regression → logs to MLflow
→ Trains Random Forest      → logs to MLflow
→ Trains XGBoost            → logs to MLflow
→ Selects best by F1-Score

--- Step 6: Preparing prediction inputs for active customers ---
→ Extracts active customers (NULL or 0 churn)
→ Applies fitted preprocessor

--- Step 7: Running predictions ---
→ Generates churn_probability and churn_predicted_class
→ Saves prediction_results.csv

--- Step 8: Running SHAP explanations ---
→ SHAP TreeExplainer on winning model
→ Extracts top_1 (retention) and top_2 (churn) drivers per customer
→ Saves dynamic_insights.csv
→ Computes mean(|SHAP|) for global importance
→ Aggregates OHE dummies back to original features
→ Saves global_feature_importance.csv and raw_global_feature_importance.csv

--- Step 9: Generating feature impact CSVs for Power BI ---
→ Reads top 3 features from global_feature_importance.csv
→ Generates smart-binned or categorical impact tables
→ Saves feature_impact_csvs/*.csv
```

---

## 10. ML Models — Training & Selection

### Model Configurations

**Logistic Regression (Baseline)**
```python
LogisticRegression(max_iter=1000, random_state=42)
```
- Linear model; interpretable but limited on non-linear churn patterns
- Serves as the lower-bound performance benchmark

**Random Forest**
```python
RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
```
- 100 decision trees, depth-limited to 12 to prevent overfitting
- Robust to feature scale differences (doesn't need standardization to function, but receives it anyway)

**XGBoost**
```python
XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=42)
```
- Gradient-boosted trees; uses `logloss` as internal evaluation metric
- `max_depth=6` is more conservative than RF's 12 to guard against overfitting on boosted ensemble

### Why F1-Score for Selection?

Customer churn is a **class-imbalanced problem** (minority: churned customers). Accuracy alone would be misleading — a model that always predicts "no churn" would score ~80%+ accuracy but identify zero churners. F1-Score (harmonic mean of precision and recall) penalizes both false positives and false negatives, making it the right metric for this use case.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Actual Run Results

| Model | Validation F1-Score |
|---|---|
| Logistic Regression | 0.5069 |
| **Random Forest** ✅ | **0.5997** |
| XGBoost | 0.5884 |

**Winner: Random Forest** with F1 = **0.5997**

---

## 11. SHAP Explainability Engine

SHAP (SHapley Additive exPlanations) provides **consistent, mathematically grounded** feature attribution. Each SHAP value represents how much a specific feature value pushed a prediction *above or below* the model's base rate.

### How SHAP is Applied Here

- **Explainer type:** `shap.TreeExplainer` (used for Random Forest and XGBoost) — this is the fastest and most exact explainer for tree-based models.
- **Input:** `X_active_processed` — the preprocessed feature matrix for all active customers
- **Output:** `shap_values` matrix of shape `(n_customers, n_features)`

### Per-Customer Driver Extraction

For each customer, the SHAP values are sorted by magnitude:
- `top_1` = most *negative* SHAP value → strongest retention signal
- `top_2` = most *positive* SHAP value → strongest churn signal

```python
sorted_indices = np.argsort(customer_shap)[::-1]  # descending by value
top_1_idx = sorted_indices[-1]   # most negative (retention)
top_2_idx = sorted_indices[0]    # most positive (churn)
```

### Global Feature Importance

```python
global_importances = np.mean(np.abs(shap_values), axis=0)
```

This gives the **mean absolute SHAP value** per feature — a stable measure of how much each feature contributes to predictions on average across the entire customer base.

OHE dummy columns (e.g., `gender_Male`, `gender_Female`) are then aggregated back to their parent feature (`gender`) by summing their importances.

---

## 12. Smart Binning Algorithm

The `generate_feature_impact_csvs.py` module contains a sophisticated numerical binning system designed to make charts in Power BI look clean and meaningful regardless of the feature's scale or distribution.

### Decision Rules

```
value_range = max(values) - min(values)

if value_range == 0:     → 1 bin  (single-value feature)
if value_range < 1:      → up to 3 bins (percentile-spaced at P33/P67)
if 1 ≤ range ≤ 10:       → NO bins (each distinct integer = its own row)
if 10 < range ≤ 24:      → 6 bins (integer-aligned)
if range > 24:           → 8 bins (integer-aligned)
```

### Outlier Clamping

Interior bin edges are placed within the **P2–P98 range** (not the raw min/max) when outlier tails are large. This prevents one extreme outlier from stretching all bins into useless wide ranges.

Clamping is activated when:
```
outlier_tail > 0.20 × (P98 - P2)
```

### Age Example (range > 24 → 8 bins)

```
value_bin | retention_impact | churn_impact
19–25     |   -38.47         |    0.00    ← young customers strongly retained
25–32     |  -225.07         |    0.00
32–38     |  -313.94         |    0.00    ← strongest retention age group
38–44     |   -83.38         |    8.46    ← transition zone
44–51     |     0.00         |  111.40    ← strongest churn age group
51–57     |     0.00         |   43.33
57–64     |     0.00         |   28.49
64+       |     0.00         |   13.13    ← older customers also churn more
```

**Insight:** Customers aged **44–51** are by far the highest-risk churn group. Customers aged **32–38** are the most strongly retained.

---

## 13. Output Files & Artifacts

### `data_quality_metrics.csv`

Single-row health report from the DB validation procedure:

```
total_loaded | total_rejected | total_duplicates | total_clean | salary_missing_pct
10,150       | 498            | 145              | 9,507       | 5.96%
```

- **4.9% rejection rate** (498 bad records)
- **1.4% duplicate rate** (145 duplicate customer IDs)
- **5.96% of records** have a missing `estimated_salary`

### `prediction_results.csv`

Full row export for all **7,559 active customers**, including all original feature columns plus:
- `churn_probability` — float between 0.0 and 1.0
- `churn_predicted_class` — binary 0 or 1

### `dynamic_insights.csv`

One row per active customer with SHAP-derived explanations:

| Column | Description |
|---|---|
| `customer_id` | Customer identifier |
| `churn_probability` | Model's predicted probability of churn |
| `top_1_name` | Name of the strongest retention driver feature |
| `top_1_value` | Original (unscaled) value of that feature |
| `top_1_impact` | SHAP value (negative = pushes toward retention) |
| `top_2_name` | Name of the strongest churn driver feature |
| `top_2_value` | Original (unscaled) value of that feature |
| `top_2_impact` | SHAP value (positive = pushes toward churn) |

### `global_feature_importance.csv`

Ranked list of 14 original features by mean absolute SHAP value:

| Feature | Global Importance |
|---|---|
| `age` | 0.127003 ← **#1 driver** |
| `products_number` | 0.081295 |
| `gender` | 0.061789 |
| `country` | 0.058613 |
| `active_member` | 0.056108 |
| `cidade` | 0.044060 |
| `limite_credito` | 0.025957 |
| `balance` | 0.017512 |
| `credit_utilization` | 0.006340 |
| `tenure` | 0.005906 |
| `credit_score` | 0.005881 |
| `estimated_salary` | 0.005588 |
| `credit_card` | 0.002014 |
| `salary_missing_flag` | 0.000993 |

### `raw_global_feature_importance.csv`

Same as above but shows each OHE dummy column separately (e.g., `gender_Male: 0.031965`, `gender_Female: 0.029824`). Useful for debugging and understanding directional effects.

### `feature_impact_csvs/`

#### `age_impact.csv` — 8 bins
| value_bin | retention_impact | num_retention_customers | churn_impact | num_churn_customers |
|---|---|---|---|---|
| 19–25 | -38.47 | 235 | 0.0 | 0 |
| 25–32 | -225.07 | 1343 | 0.0 | 0 |
| 32–38 | -313.94 | 2043 | 0.0 | 0 |
| 38–44 | -83.38 | 782 | 8.46 | 155 |
| 44–51 | 0.0 | 0 | 111.40 | 953 |
| 51–57 | 0.0 | 0 | 43.33 | 255 |
| 57–64 | 0.0 | 0 | 28.49 | 204 |
| 64+ | 0.0 | 0 | 13.13 | 214 |

#### `gender_impact.csv` — Categorical
| value | retention_impact | num_retention_customers | churn_impact | num_churn_customers |
|---|---|---|---|---|
| Female | 0.0 | 0 | 2.67 | 154 |
| Male | -0.77 | 19 | 0.0 | 0 |

**Insight:** Female customers are slightly more likely to churn (2.67 churn impact), while male customers show minimal retention signal.

#### `products_number_impact.csv` — Exact values (range = 3)
| value | retention_impact | num_retention_customers | churn_impact | num_churn_customers |
|---|---|---|---|---|
| 1 | 0.0 | 0 | 47.71 | 1183 |
| 2 | -268.48 | 1970 | 0.0 | 0 |
| 3 | 0.0 | 0 | 8.30 | 44 |

**Insight:** Customers with **2 products** are strongly retained (largest retention pool). Customers with **1 product** show the highest churn risk.

---

## 14. Actual Execution Results

From `pipeline_execution_log.txt` (captured from a real run on 2026-06-22):

```
Starting Bank Customer Churn Pipeline Execution

Step 1: sp_clean_and_validate_data() → Success
Step 2: data_quality_metrics.csv → Saved
Step 3: Retrieved 9,507 records from vw_Analytics_Features
Step 4: Original shape (7,605, 39) → Resampled (12,094, 39)  [SMOTE applied]
Step 5:
  - Logistic_Regression → F1: 0.5069
  - Random_Forest       → F1: 0.5997  ← WINNER
  - XGBoost             → F1: 0.5884
Step 6: No NULL churn records → Fallback to churn=0 customers
Step 7: prediction_results.csv → 7,559 customers
Step 8: dynamic_insights.csv → Exported
Step 9: 3 CSVs exported to feature_impact_csvs/

Pipeline Execution Completed Successfully!
```

**MLflow warnings** (non-blocking): `artifact_path` deprecation in newer MLflow versions (use `name` instead).

---

## 15. MLflow Experiment Tracking

All training runs are logged to the `Bank_Customer_Churn_Prediction` MLflow experiment.

**Backend:** SQLite (`mlflow.db`) — no server required, local file-based tracking.

**To view the MLflow UI:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open: `http://localhost:5000`

**Each run records:**
- `run_name`: Model name (e.g., `Random_Forest`)
- **Parameters:** all hyperparameters from `model.get_params()`
- **Metrics:** `validation_f1_score`, `validation_precision`, `validation_recall`, `validation_accuracy`
- **Artifacts:** serialized model (`.pkl` for sklearn, native format for XGBoost)

`mlruns/` directory contains all run artifacts. `mlflow.db` contains the experiment metadata in SQLite format.

---

## 16. Power BI Dashboard Architecture

The pipeline outputs are structured specifically to feed a **3-page Power BI dashboard**.

### Page 1: Data Quality & Health

**Data source:** `data_quality_metrics.csv`

| Visual | Type | Data |
|---|---|---|
| Total Records Received | KPI Card | `total_loaded` |
| Validated Records | KPI Card | `total_clean` |
| Rejection Rate (%) | KPI Card | `total_rejected / total_loaded * 100` |
| Duplicate Count | KPI Card | `total_duplicates` |
| Data Health Breakdown | Pie Chart | Clean / Rejected / Duplicated split |
| Missing Salary % | Bar / Gauge | `salary_missing_pct` |

### Page 2: Core Business Analytics

**Data source:** `prediction_results.csv`

| Visual | Type | X-Axis | Y-Axis |
|---|---|---|---|
| Churn Rate by Tenure | Line Chart | `tenure` | Churn Rate |
| Churn Rate by Products | Column Chart | `products_number` | Churn Rate |
| Churn by Geography | Map | `country` | Churn Rate / Avg Balance |
| Churn by Age Band | Bar Chart | `age_band` | Churn Rate |

### Page 3: Dynamic ML Insights (Key Page)

**Data sources:** `dynamic_insights.csv`, `feature_impact_csvs/*.csv`

This page uses **DAX Dynamic Measures** to self-update its titles and chart labels based on whatever features the current model identifies as most important. The dashboard adapts automatically every time the pipeline is re-run on new data.

**DAX measure for dynamic visual title:**
```dax
Visual_1_Title = 
    "Churn Probability vs " & 
    FIRSTNONBLANK(dynamic_insights[top_1_name], "Primary Driver")

Visual_2_Title = 
    "Churn Rate by " & 
    FIRSTNONBLANK(dynamic_insights[top_2_name], "Secondary Driver")
```

**Visuals on this page:**
- Scatter plot: `top_1_value` vs `churn_probability`
- Bar chart: `top_2_value` vs average churn rate
- Feature impact cards using `feature_impact_csvs/`

---

## 17. Feature Importance Results

### Top 3 Most Important Features (SHAP — Global)

1. **`age`** (importance: 0.127) — By far the strongest predictor. Middle-aged customers (44–57) churn most.
2. **`products_number`** (importance: 0.081) — 2-product customers retained; 1-product customers at highest churn risk.
3. **`gender`** (importance: 0.062) — Female customers show higher churn tendency in this dataset.

### Notable Findings

- **`active_member`** (0.056) is the 5th most important feature — customers who are not actively using banking services are significantly more likely to leave.
- **`credit_utilization`** (0.006) and **`credit_score`** (0.006) have surprisingly low SHAP importance, suggesting that financial risk metrics are less predictive of churn than demographic and behavioral factors.
- **`salary_missing_flag`** (0.001) has minimal impact, validating that median imputation + missing flag strategy works correctly (the missingness itself carries little signal).

---

## 18. How to Run

### Prerequisites

1. **MySQL 8.0+** running on `localhost:3306`
2. **Database `creditcard`** must exist
3. **Python 3.10+** with virtualenv

### Setup Steps

```bash
# 1. Clone / navigate to the project
cd BankProject/

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# or: venv\Scripts\activate     # Windows

# 3. Install dependencies (see Section 19)
pip install -r requirements.txt

# 4. Set up the MySQL database (run once)
mysql -u root -p creditcard < 01_data_quality_setup.sql
mysql -u root -p creditcard < 02_analytical_views.sql

# 5. Load raw CSV data into MySQL (run once)
python import_csv.py

# 6. Run the full pipeline (every time you want fresh predictions)
python run_pipeline.py
```

### What Gets Generated After `run_pipeline.py`

```
✅ data_quality_metrics.csv         — Data health report
✅ prediction_results.csv           — Churn scores for all active customers
✅ dynamic_insights.csv             — Per-customer SHAP explanations
✅ global_feature_importance.csv    — Ranked global feature drivers
✅ raw_global_feature_importance.csv
✅ feature_impact_csvs/
      age_impact.csv
      products_number_impact.csv
      gender_impact.csv
```

---

## 19. Dependencies & Installation

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost shap mlflow pymysql sqlalchemy
```

| Package | Version (recommended) | Purpose |
|---|---|---|
| `pandas` | ≥1.5 | DataFrame operations |
| `numpy` | ≥1.23 | Numerical arrays |
| `scikit-learn` | ≥1.2 | ML models, preprocessing, metrics |
| `imbalanced-learn` | ≥0.10 | SMOTE |
| `xgboost` | ≥1.7 | XGBoost classifier |
| `shap` | ≥0.41 | SHAP explainability |
| `mlflow` | ≥2.0 | Experiment tracking |
| `pymysql` | ≥1.0 | MySQL raw connection |
| `sqlalchemy` | ≥1.4 | ORM engine for `pd.read_sql()` |

---

## 20. Configuration Reference

All configurable parameters are hardcoded in their respective files. Below is a consolidated reference for easy modification:

### Database (src/db_connector.py)

| Parameter | Default | Description |
|---|---|---|
| `DB_HOST` | `"localhost"` | MySQL server host |
| `DB_PORT` | `3306` | MySQL server port |
| `DB_USER` | `"root"` | MySQL username |
| `DB_PASSWORD` | `"hriday7580"` | MySQL password |
| `DB_NAME` | `"creditcard"` | Target database name |

> ⚠️ **Security Note:** Credentials are hardcoded. For production, use environment variables or a secrets manager.

### Model Training (src/train.py)

| Parameter | Default | Description |
|---|---|---|
| RF `n_estimators` | `100` | Number of trees in Random Forest |
| RF `max_depth` | `12` | Max depth per tree |
| XGB `n_estimators` | `100` | Boosting rounds |
| XGB `max_depth` | `6` | Max depth per XGB tree |
| XGB `learning_rate` | `0.1` | Step size shrinkage |
| LR `max_iter` | `1000` | Max solver iterations |
| `random_state` | `42` | Global random seed (all models) |

### Preprocessing (src/preprocessing.py)

| Parameter | Default | Description |
|---|---|---|
| SMOTE `random_state` | `42` | SMOTE reproducibility |
| Train/Val split | `80/20` | `test_size=0.2` |
| Split `stratify` | `y` | Maintains class distribution |
| Imputation strategy | `'median'` | For missing `estimated_salary` |

### Feature Impact Generator (generate_feature_impact_csvs.py)

| Parameter | Default | Description |
|---|---|---|
| `top_n` | `3` | Number of top features to generate CSVs for |
| `GLOBAL_IMPORTANCE_FILE` | `"global_feature_importance.csv"` | Input importance file |
| `DYNAMIC_INSIGHTS_FILE` | `"dynamic_insights.csv"` | Input SHAP insights file |
| `OUTPUT_DIR` | `"feature_impact_csvs"` | Output directory for impact CSVs |

---

## 📊 Quick Summary Stats

| Metric | Value |
|---|---|
| Raw records loaded | 10,150 |
| Records rejected (validation) | 498 (4.9%) |
| Records deduplicated | 145 (1.4%) |
| Clean records used | 9,507 |
| Features used for training | 39 (post-OHE) |
| Training samples (after SMOTE) | 12,094 |
| Validation samples | 1,902 |
| Active customers predicted | 7,559 |
| Best model | Random Forest |
| Best validation F1-Score | 0.5997 |
| #1 churn driver | `age` (SHAP = 0.127) |
| #2 churn driver | `products_number` (SHAP = 0.081) |

---

*Generated from a complete analysis of all source files, SQL scripts, output artifacts, and execution logs.*

# generate_feature_impact_csvs.py
# Reads the top 3 features from global_feature_importance.csv dynamically.
# For each feature, produces a CSV showing retention vs churn impact by value/bin,
# using customer-level SHAP data from dynamic_insights.csv.
#
# Output columns (categorical):  value, retention_impact, num_retention_customers, churn_impact, num_churn_customers
# Output columns (numerical):    value_bin, retention_impact, num_retention_customers, churn_impact, num_churn_customers
#
# Output files are saved to: feature_impact_csvs/<feature_name>_impact.csv
#
# ── Smart Binning Rules Applied ────────────────────────────────────────────────
#
#  MAIN RULE  (range = max − min)
#    • range > 24        → 8  bins  (smart integer-aligned bins)
#    • 10 < range ≤ 24   → 6  bins  (smart integer-aligned bins)
#    • 1  ≤ range ≤ 10   → NO bins; each distinct integer value = its own row
#    • range < 1         → 3  bins  (percentile-spaced, decimal labels)
#    • range = 0         → 1  bin   (single-value feature)
#    Interior cut-points are rounded to the nearest integer so every bin
#    boundary is a whole number:
#      first bin  →  true_min – first_integer    (e.g. "18–28")
#      middle bins → integer – integer            (e.g. "28–37")
#      last  bin  →  last_integer+               (e.g. "80+")
#    Bin sizes are allowed to vary to honour integer boundaries.
#
#  SEMI-MAIN RULE  (outlier-aware edge placement)
#    Interior edges are spaced within the *effective* range [P2, P98] of
#    the data, not the raw [min, max].  This prevents a single extreme
#    outlier from stretching all bins so wide that most customers crowd
#    into just one or two of them.  The first and last bins automatically
#    absorb any values that lie outside the effective range, so no data
#    is ever lost.  Outlier clamping is only activated when the outlier
#    tail exceeds 20 % of the core (P2–P98) spread.
#
# ──────────────────────────────────────────────────────────────────────────────

import math
import os
import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────

OHE_PREFIXES   = ["country_", "cidade_", "gender_", "age_band_"]
OHE_BASE_NAMES = [p[:-1] for p in OHE_PREFIXES]   # ["country", "cidade", "gender", "age_band"]

GLOBAL_IMPORTANCE_FILE = "global_feature_importance.csv"
DYNAMIC_INSIGHTS_FILE  = "dynamic_insights.csv"
OUTPUT_DIR             = "feature_impact_csvs"


# ── Smart Binning ──────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    """Format an edge: integer representation when whole, compact float otherwise."""
    if v == math.floor(v):
        return str(int(v))
    return f"{v:.4g}"


def compute_smart_bins(values_series: pd.Series):
    """
    Compute bin edges and human-readable labels for a numerical feature.

    Parameters
    ----------
    values_series : pd.Series
        All observed values for the feature (combined from top_1 and top_2).

    Returns
    -------
    edges  : list[float] – cut points for pd.cut (length = num_bins + 1)
    labels : list[str]   – one label per bin
    """
    values = values_series.dropna()
    if values.empty:
        return None, []

    true_min   = float(values.min())
    true_max   = float(values.max())
    value_range = true_max - true_min

    # ── Edge case: all values identical ───────────────────────────────────
    if value_range == 0:
        eps = max(abs(true_min) * 1e-6, 1e-9)
        return [true_min - eps, true_max + eps], [_fmt(true_min)]

    # ── range < 1: very tight / fractional range ───────────────────────────
    # Rule: use 3 fixed bins spaced at the 33rd and 67th percentiles so
    # the bins reflect where values actually concentrate, not equal width.
    if value_range < 1:
        p33 = float(np.percentile(values, 33))
        p67 = float(np.percentile(values, 67))

        # Keep only interior cuts that are strictly inside (true_min, true_max)
        interior = sorted({e for e in [p33, p67] if true_min < e < true_max})
        all_edges = [true_min] + interior + [true_max]

        labels = []
        n = len(all_edges) - 1
        for i in range(n):
            lo, hi = all_edges[i], all_edges[i + 1]
            labels.append(f"{_fmt(lo)}+" if i == n - 1 else f"{_fmt(lo)}–{_fmt(hi)}")
        return all_edges, labels

    # ── range ≥ 1: determine target number of bins ─────────────────────────
    # NOTE: range 1–10 is handled BEFORE this function (exact-value rows).
    # This function only bins when range > 10.
    if value_range > 24:
        num_bins = 8
    else:                          # 10 < range ≤ 24
        num_bins = 6

    # ── Outlier-aware effective inner range ────────────────────────────────
    # Use P2 / P98 to compute interior edges only when the outlier tail is
    # meaningfully large (> 20 % of the P2–P98 core spread).
    p2  = float(np.percentile(values, 2))
    p98 = float(np.percentile(values, 98))
    core_spread = max(p98 - p2, 1e-9)

    inner_lo = math.floor(p2)  if (p2  - true_min) > 0.20 * core_spread else math.floor(true_min)
    inner_hi = math.ceil(p98)  if (true_max - p98)  > 0.20 * core_spread else math.ceil(true_max)

    # Safety: ensure valid inner range
    if inner_hi <= inner_lo:
        inner_lo = math.floor(true_min)
        inner_hi = math.ceil(true_max)

    # ── Generate num_bins − 1 integer-aligned interior cut points ──────────
    # Spread them evenly across [inner_lo, inner_hi] then round to integers.
    raw_interior = np.linspace(inner_lo, inner_hi, num_bins + 1)[1:-1]
    seen         = set()
    int_interior = []
    for v in raw_interior:
        r = round(v)
        if true_min < r < true_max and r not in seen:
            int_interior.append(r)
            seen.add(r)
    int_interior.sort()

    all_edges = [true_min] + int_interior + [true_max]

    # ── Build labels ───────────────────────────────────────────────────────
    labels = []
    n = len(all_edges) - 1
    for i in range(n):
        lo, hi = all_edges[i], all_edges[i + 1]
        labels.append(f"{_fmt(lo)}+" if i == n - 1 else f"{_fmt(lo)}–{_fmt(hi)}")

    return all_edges, labels


# ── Feature table builders ─────────────────────────────────────────────────────

def is_categorical(feature_name: str) -> bool:
    return feature_name in OHE_BASE_NAMES


def build_categorical_csv(feature_name: str, insights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorical (OHE) feature impact table.

    Retention side → top_1_name starts with '<feature>_' AND top_1_value == 1
    Churn side     → top_2_name starts with '<feature>_' AND top_2_value == 1
    """
    prefix = feature_name + "_"

    # Retention (top_1)
    ret_mask = (
        insights_df["top_1_name"].str.startswith(prefix, na=False) &
        (insights_df["top_1_value"] == 1.0)
    )
    ret_df = insights_df[ret_mask].copy()
    ret_df["category"] = ret_df["top_1_name"].str[len(prefix):]
    ret_grouped = (
        ret_df.groupby("category")
        .agg(retention_impact=("top_1_impact", "sum"),
             num_retention_customers=("top_1_impact", "count"))
        .reset_index()
        .rename(columns={"category": "value"})
    )

    # Churn (top_2)
    churn_mask = (
        insights_df["top_2_name"].str.startswith(prefix, na=False) &
        (insights_df["top_2_value"] == 1.0)
    )
    churn_df = insights_df[churn_mask].copy()
    churn_df["category"] = churn_df["top_2_name"].str[len(prefix):]
    churn_grouped = (
        churn_df.groupby("category")
        .agg(churn_impact=("top_2_impact", "sum"),
             num_churn_customers=("top_2_impact", "count"))
        .reset_index()
        .rename(columns={"category": "value"})
    )

    result = pd.merge(ret_grouped, churn_grouped, on="value", how="outer").fillna(0)
    result["retention_impact"]        = result["retention_impact"].round(6)
    result["churn_impact"]            = result["churn_impact"].round(6)
    result["num_retention_customers"] = result["num_retention_customers"].astype(int)
    result["num_churn_customers"]     = result["num_churn_customers"].astype(int)
    return result[["value", "retention_impact", "num_retention_customers",
                   "churn_impact", "num_churn_customers"]]


def build_exact_value_csv(feature_name: str, insights_df: pd.DataFrame, true_vals: pd.Series = None) -> pd.DataFrame:
    """
    Small-range numerical feature (1 ≤ max−min ≤ 10).
    Each distinct integer value gets its own row — no binning at all.
    Structure mirrors the categorical table exactly.
    """
    ret_df = insights_df[insights_df["top_1_name"] == feature_name][
        ["top_1_value", "top_1_impact"]].copy()
    ret_df.columns = ["value", "impact"]

    churn_df = insights_df[insights_df["top_2_name"] == feature_name][
        ["top_2_value", "top_2_impact"]].copy()
    churn_df.columns = ["value", "impact"]

    # Round to nearest integer so "2.0" and "2" group together
    ret_df["value"]   = ret_df["value"].dropna().round().astype(int)
    churn_df["value"] = churn_df["value"].dropna().round().astype(int)

    ret_grouped = (
        ret_df.dropna(subset=["value"]).groupby("value")
        .agg(retention_impact=("impact", "sum"),
             num_retention_customers=("impact", "count"))
        .reset_index()
    )
    churn_grouped = (
        churn_df.dropna(subset=["value"]).groupby("value")
        .agg(churn_impact=("impact", "sum"),
             num_churn_customers=("impact", "count"))
        .reset_index()
    )

    result = pd.merge(ret_grouped, churn_grouped, on="value", how="outer").fillna(0)

    if true_vals is not None and not true_vals.empty:
        t_min = int(round(true_vals.min()))
        t_max = int(round(true_vals.max()))
        if t_max >= t_min:
            all_ints = pd.DataFrame({"value": range(t_min, t_max + 1)})
            result = pd.merge(all_ints, result, on="value", how="left").fillna(0)

    result["value"]                   = result["value"].astype(int)
    result["retention_impact"]        = result["retention_impact"].round(6)
    result["churn_impact"]            = result["churn_impact"].round(6)
    result["num_retention_customers"] = result["num_retention_customers"].astype(int)
    result["num_churn_customers"]     = result["num_churn_customers"].astype(int)
    return result.sort_values("value").reset_index(drop=True)[
        ["value", "retention_impact", "num_retention_customers",
         "churn_impact", "num_churn_customers"]]


def build_numerical_csv(feature_name: str, insights_df: pd.DataFrame, true_vals: pd.Series = None) -> pd.DataFrame:
    """
    Numerical feature impact table with smart bins.

    Retention side → top_1_name == feature_name  (most-negative driver)
    Churn side     → top_2_name == feature_name  (most-positive driver)

    Bin edges are computed from the union of both sides' values so that
    both sides always use an identical, consistent set of bin labels.
    """
    ret_df = insights_df[insights_df["top_1_name"] == feature_name][
        ["top_1_value", "top_1_impact"]].copy()
    ret_df.columns = ["value", "impact"]

    churn_df = insights_df[insights_df["top_2_name"] == feature_name][
        ["top_2_value", "top_2_impact"]].copy()
    churn_df.columns = ["value", "impact"]

    all_values = pd.concat([ret_df["value"], churn_df["value"]]).dropna()
    
    if true_vals is None or true_vals.empty:
        true_vals = all_values

    if true_vals.empty:
        print(f"  [WARNING] No data found for '{feature_name}'. Skipping.")
        return pd.DataFrame(columns=["value_bin", "retention_impact",
                                     "num_retention_customers", "churn_impact",
                                     "num_churn_customers"])

    edges, labels = compute_smart_bins(true_vals)
    if edges is None or not labels:
        return pd.DataFrame(columns=["value_bin", "retention_impact",
                                     "num_retention_customers", "churn_impact",
                                     "num_churn_customers"])

    # Slightly extend the last edge so pd.cut (right=False) captures the max value
    cut_edges = list(edges)
    cut_edges[-1] += max(abs(cut_edges[-1]) * 1e-6, 1e-9)

    def _bin_and_agg(df: pd.DataFrame, impact_col: str, count_col: str) -> pd.DataFrame:
        empty = pd.DataFrame({"value_bin": labels,
                               impact_col: 0.0,
                               count_col: 0})
        if df.empty:
            return empty
        df = df.dropna(subset=["value"]).copy()
        df["_bin_idx"] = pd.cut(df["value"], bins=cut_edges, labels=False,
                                right=False, include_lowest=True)
        df["value_bin"] = df["_bin_idx"].apply(
            lambda i: labels[int(i)] if pd.notna(i) else None
        )
        df = df.dropna(subset=["value_bin"])
        if df.empty:
            return empty
        return (
            df.groupby("value_bin")
            .agg(**{impact_col: ("impact", "sum"),
                    count_col:  ("impact", "count")})
            .reset_index()
        )

    ret_grouped   = _bin_and_agg(ret_df,   "retention_impact", "num_retention_customers")
    churn_grouped = _bin_and_agg(churn_df, "churn_impact",     "num_churn_customers")

    # Ensure all bin labels appear (fill missing with 0)
    skeleton = pd.DataFrame({"value_bin": labels})
    result   = skeleton.merge(ret_grouped,   on="value_bin", how="left")
    result   = result.merge(churn_grouped, on="value_bin", how="left")
    result   = result.fillna(0)
    result["retention_impact"]        = result["retention_impact"].round(6)
    result["churn_impact"]            = result["churn_impact"].round(6)
    result["num_retention_customers"] = result["num_retention_customers"].astype(int)
    result["num_churn_customers"]     = result["num_churn_customers"].astype(int)
    return result[["value_bin", "retention_impact", "num_retention_customers",
                   "churn_impact", "num_churn_customers"]]


# ── Main ───────────────────────────────────────────────────────────────────────

def generate_feature_impact_csvs(
    global_importance_file: str = GLOBAL_IMPORTANCE_FILE,
    dynamic_insights_file:  str = DYNAMIC_INSIGHTS_FILE,
    output_dir:             str = OUTPUT_DIR,
    top_n:                  int = 3,
) -> None:
    """
    Entry point: reads top_n features dynamically and writes one CSV per feature.
    """
    print("=" * 56)
    print("  Feature Impact CSV Generator  (smart binning)")
    print("=" * 56)

    global_df    = pd.read_csv(global_importance_file)
    top_features = global_df.head(top_n)["feature_name"].tolist()
    print(f"\nTop {top_n} features (dynamic): {top_features}")

    insights_df = pd.read_csv(dynamic_insights_file)

    prediction_file = "prediction_results.csv"
    if os.path.exists(prediction_file):
        raw_data = pd.read_csv(prediction_file)
    else:
        raw_data = None

    os.makedirs(output_dir, exist_ok=True)

    for feature in top_features:
        print(f"\n{'─' * 40}")
        print(f"  Feature : {feature}")

        if is_categorical(feature):
            print("  Type    : Categorical (OHE)")
            result_df = build_categorical_csv(feature, insights_df)
        else:
            # Show binning decision before building
            if raw_data is not None and feature in raw_data.columns:
                true_vals = raw_data[feature].dropna()
            else:
                true_vals = pd.concat([
                    insights_df[insights_df["top_1_name"] == feature]["top_1_value"],
                    insights_df[insights_df["top_2_name"] == feature]["top_2_value"],
                ]).dropna()

            if not true_vals.empty:
                r = true_vals.max() - true_vals.min()
                if r > 24:
                    mode, detail = "binned", "8 bins"
                elif r > 10:
                    mode, detail = "binned", "6 bins"
                elif r >= 1:
                    mode, detail = "exact values", f"range={r:.4g}, no bins"
                else:
                    mode, detail = "binned", "3 percentile bins (range<1)"
                print(f"  Type    : Numerical  |  {detail}  [{mode}]")

            if not true_vals.empty and (true_vals.max() - true_vals.min()) <= 10:
                result_df = build_exact_value_csv(feature, insights_df, true_vals)
            else:
                result_df = build_numerical_csv(feature, insights_df, true_vals)

        output_path = os.path.join(output_dir, f"{feature}_impact.csv")
        result_df.to_csv(output_path, index=False)
        print(f"  Saved   : {output_path}  ({len(result_df)} rows)")
        print(result_df.to_string(index=False))

    print(f"\n{'=' * 56}")
    print(f"  Done! {top_n} CSVs saved to '{output_dir}/'")
    print("=" * 56)


if __name__ == "__main__":
    generate_feature_impact_csvs()

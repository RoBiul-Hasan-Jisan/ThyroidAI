"""
Computes JSON-serializable EDA aggregates from the raw dataset for the
frontend's interactive Recharts-based analytics dashboard.

Enhanced with additional visualizations:
- Correlation matrix (top features)
- Feature importance ranking
- Risk factor combination analysis
- Age statistical summary
- Categorical feature summaries
- Missing value analysis
- Class imbalance metrics
- Multi-feature cross-tabs
"""
import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "Thyroid_Diff.csv")

_df_cache = None


def _get_df():
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv(DATA_PATH)
    return _df_cache


def _crosstab_pct(df, col):
    """Calculate percentage crosstab for a categorical column."""
    ct = pd.crosstab(df[col], df["Recurred"], normalize="index") * 100
    ct = ct.reindex(columns=["No", "Yes"], fill_value=0)
    out = []
    for idx, row in ct.iterrows():
        total = len(df[df[col] == idx])
        out.append({
            "category": str(idx),
            "No": round(row["No"], 1),
            "Yes": round(row["Yes"], 1),
            "total": int(total),
            "percentage": round((total / len(df)) * 100, 1)
        })
    return out


def _get_encoded_df(df):
    """Get label-encoded version for correlation analysis."""
    enc_df = df.copy()
    for c in enc_df.columns:
        if not pd.api.types.is_numeric_dtype(enc_df[c]):
            enc_df[c] = LabelEncoder().fit_transform(enc_df[c].astype(str))
    return enc_df


def _get_correlation_analysis(df):
    """Compute correlation analysis with target."""
    enc_df = _get_encoded_df(df)
    corr_matrix = enc_df.corr()
    target_corr = corr_matrix["Recurred"].drop("Recurred").sort_values(ascending=False)
    
    return {
        "top_positive": [
            {"feature": k, "correlation": round(v, 3)}
            for k, v in target_corr.nlargest(5).items()
        ],
        "top_negative": [
            {"feature": k, "correlation": round(v, 3)}
            for k, v in target_corr.nsmallest(5).items()
        ],
        "all_features": [
            {"feature": k, "correlation": round(v, 3), "abs_correlation": round(abs(v), 3)}
            for k, v in target_corr.items()
        ]
    }


def _get_age_analysis(df):
    """Detailed age analysis with statistics."""
    age_no = df[df["Recurred"] == "No"]["Age"]
    age_yes = df[df["Recurred"] == "Yes"]["Age"]
    
    return {
        "overall": {
            "mean": round(df["Age"].mean(), 2),
            "median": round(df["Age"].median(), 2),
            "std": round(df["Age"].std(), 2),
            "min": int(df["Age"].min()),
            "max": int(df["Age"].max()),
            "q25": round(df["Age"].quantile(0.25), 2),
            "q75": round(df["Age"].quantile(0.75), 2),
        },
        "by_recurrence": {
            "No": {
                "mean": round(age_no.mean(), 2),
                "median": round(age_no.median(), 2),
                "std": round(age_no.std(), 2),
                "count": int(len(age_no))
            },
            "Yes": {
                "mean": round(age_yes.mean(), 2),
                "median": round(age_yes.median(), 2),
                "std": round(age_yes.std(), 2),
                "count": int(len(age_yes))
            }
        }
    }


def _get_feature_summary(df):
    """Get summary statistics for all features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    summary = {
        "numeric": [],
        "categorical": []
    }
    
    for col in numeric_cols:
        if col != "Recurred":
            summary["numeric"].append({
                "feature": col,
                "mean": round(df[col].mean(), 2),
                "median": round(df[col].median(), 2),
                "std": round(df[col].std(), 2),
                "min": int(df[col].min()),
                "max": int(df[col].max()),
                "missing": int(df[col].isnull().sum()),
                "missing_pct": round((df[col].isnull().sum() / len(df)) * 100, 1)
            })
    
    for col in categorical_cols:
        if col != "Recurred":
            value_counts = df[col].value_counts()
            summary["categorical"].append({
                "feature": col,
                "unique_count": int(df[col].nunique()),
                "most_common": str(value_counts.index[0]),
                "most_common_count": int(value_counts.iloc[0]),
                "most_common_pct": round((value_counts.iloc[0] / len(df)) * 100, 1),
                "missing": int(df[col].isnull().sum()),
                "missing_pct": round((df[col].isnull().sum() / len(df)) * 100, 1),
                "value_counts": [
                    {"value": str(k), "count": int(v), "pct": round((v / len(df)) * 100, 1)}
                    for k, v in value_counts.items()
                ]
            })
    
    return summary


def _get_risk_combination_analysis(df):
    """Analyze combinations of risk factors."""
    risk_combinations = []
    
    # Risk and Stage combination
    risk_stage = pd.crosstab([df["Risk"], df["Stage"]], df["Recurred"])
    for (risk, stage), counts in risk_stage.iterrows():
        total = counts["No"] + counts["Yes"]
        if total > 0:
            risk_combinations.append({
                "risk": str(risk),
                "stage": str(stage),
                "No": int(counts["No"]),
                "Yes": int(counts["Yes"]),
                "total": int(total),
                "recurrence_rate": round((counts["Yes"] / total) * 100, 1)
            })
    
    # T and N combination
    tn_combination = pd.crosstab([df["T"], df["N"]], df["Recurred"])
    tn_data = []
    for (t, n), counts in tn_combination.iterrows():
        total = counts["No"] + counts["Yes"]
        if total > 5:  # Only include combinations with sufficient samples
            tn_data.append({
                "t_stage": str(t),
                "n_stage": str(n),
                "No": int(counts["No"]),
                "Yes": int(counts["Yes"]),
                "total": int(total),
                "recurrence_rate": round((counts["Yes"] / total) * 100, 1)
            })
    
    return {
        "risk_stage_combination": risk_combinations,
        "tn_combination": tn_data
    }


def _get_class_imbalance_metrics(df):
    """Calculate class imbalance metrics."""
    counts = df["Recurred"].value_counts()
    no_count = counts.get("No", 0)
    yes_count = counts.get("Yes", 0)
    total = len(df)
    ratio = no_count / yes_count if yes_count > 0 else float('inf')
    
    return {
        "no_count": int(no_count),
        "yes_count": int(yes_count),
        "total": int(total),
        "no_percentage": round((no_count / total) * 100, 1),
        "yes_percentage": round((yes_count / total) * 100, 1),
        "imbalance_ratio": round(ratio, 2),
        "imbalance_severity": (
            "severe" if ratio > 5
            else "moderate" if ratio > 2
            else "balanced"
        ),
        "recommendation": (
            "Use SMOTE or class weights" if ratio > 3
            else "Use balanced class weights" if ratio > 1.5
            else "No special handling needed"
        )
    }


def _get_missing_value_analysis(df):
    """Analyze missing values in the dataset."""
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    
    if len(missing_data) > 0:
        return {
            "has_missing": True,
            "total_missing": int(missing_data.sum()),
            "total_missing_pct": round((missing_data.sum() / (df.shape[0] * df.shape[1])) * 100, 2),
            "features": [
                {
                    "feature": k,
                    "count": int(v),
                    "percentage": round((v / df.shape[0]) * 100, 1)
                }
                for k, v in missing_data.items()
            ]
        }
    else:
        return {
            "has_missing": False,
            "total_missing": 0,
            "total_missing_pct": 0,
            "features": []
        }


def _get_top_features_distribution(df):
    """Get distribution of top features by recurrence."""
    enc_df = _get_encoded_df(df)
    corr = enc_df.corr()["Recurred"].drop("Recurred").abs().sort_values(ascending=False)
    top_features = corr.head(6).index.tolist()
    
    # Get numeric features from top
    numeric_top = [f for f in top_features if f in df.select_dtypes(include=[np.number]).columns]
    categorical_top = [f for f in top_features if f in df.select_dtypes(include=['object']).columns]
    
    result = {
        "numeric_features": [],
        "categorical_features": []
    }
    
    # Numeric feature distribution
    for feature in numeric_top[:3]:
        data_no = df[df["Recurred"] == "No"][feature].dropna()
        data_yes = df[df["Recurred"] == "Yes"][feature].dropna()
        result["numeric_features"].append({
            "feature": feature,
            "no_stats": {
                "mean": round(data_no.mean(), 2),
                "median": round(data_no.median(), 2),
                "std": round(data_no.std(), 2),
                "count": int(len(data_no))
            },
            "yes_stats": {
                "mean": round(data_yes.mean(), 2),
                "median": round(data_yes.median(), 2),
                "std": round(data_yes.std(), 2),
                "count": int(len(data_yes))
            }
        })
    
    # Categorical feature distribution
    for feature in categorical_top[:3]:
        ct = pd.crosstab(df[feature], df["Recurred"], normalize="index") * 100
        ct = ct.reindex(columns=["No", "Yes"], fill_value=0)
        result["categorical_features"].append({
            "feature": feature,
            "data": [
                {
                    "category": str(idx),
                    "No": round(row["No"], 1),
                    "Yes": round(row["Yes"], 1)
                }
                for idx, row in ct.iterrows()
            ]
        })
    
    return result


def get_analytics():
    """Main function to compute all analytics for the frontend."""
    df = _get_df()
    total = len(df)

    # Basic target distribution
    target_counts = df["Recurred"].value_counts()
    target_distribution = [
        {"name": k, "value": int(v), "percentage": round((v / total) * 100, 1)}
        for k, v in target_counts.items()
    ]

    # Age histogram (5-year bins) split by recurrence
    bins = list(range(15, 90, 5))
    df_binned = df.copy()
    df_binned["age_bin"] = pd.cut(df_binned["Age"], bins=bins, right=False)
    age_hist = []
    for bin_range, group in df_binned.groupby("age_bin", observed=True):
        no_count = (group["Recurred"] == "No").sum()
        yes_count = (group["Recurred"] == "Yes").sum()
        total_count = no_count + yes_count
        age_hist.append({
            "bin": f"{int(bin_range.left)}-{int(bin_range.right)}",
            "No": int(no_count),
            "Yes": int(yes_count),
            "total": int(total_count),
            "recurrence_rate": round((yes_count / total_count) * 100, 1) if total_count > 0 else 0
        })

    # Return complete analytics with all visualizations
    return {
        # Dataset overview
        "n_samples": int(df.shape[0]),
        "n_features": int(df.shape[1] - 1),
        
        # Target distribution
        "target_distribution": target_distribution,
        "class_imbalance": _get_class_imbalance_metrics(df),
        
        # Age analysis
        "age_histogram": age_hist,
        "age_analysis": _get_age_analysis(df),
        
        # Feature summaries
        "feature_summary": _get_feature_summary(df),
        "missing_values": _get_missing_value_analysis(df),
        
        # Correlation analysis
        "correlation_analysis": _get_correlation_analysis(df),
        
        # Categorical feature crosstabs (existing)
        "gender_recurrence": _crosstab_pct(df, "Gender"),
        "smoking_recurrence": _crosstab_pct(df, "Smoking"),
        "risk_recurrence": _crosstab_pct(df, "Risk"),
        "stage_recurrence": _crosstab_pct(df, "Stage"),
        "response_recurrence": _crosstab_pct(df, "Response"),
        "t_stage_recurrence": _crosstab_pct(df, "T"),
        "n_stage_recurrence": _crosstab_pct(df, "N"),
        "m_stage_recurrence": _crosstab_pct(df, "M"),
        "adenopathy_recurrence": _crosstab_pct(df, "Adenopathy"),
        "focality_recurrence": _crosstab_pct(df, "Focality"),
        "pathology_recurrence": _crosstab_pct(df, "Pathology"),
        "thyroid_function_recurrence": _crosstab_pct(df, "Thyroid Function"),
        "physical_examination_recurrence": _crosstab_pct(df, "Physical Examination"),
        
        # Advanced analysis
        "risk_combination": _get_risk_combination_analysis(df),
        "top_features_distribution": _get_top_features_distribution(df),
        
        # Feature type information
        "feature_types": {
            "numeric": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical": df.select_dtypes(include=['object']).columns.tolist()
        }
    }

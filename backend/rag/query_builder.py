"""
Turns (patient profile + ML prediction + SHAP factors) into a small set of
short, medically-scoped retrieval queries.

Only clinically meaningful concept phrases go into a query -- we never dump
raw patient JSON, free-text identifiers, or unrelated fields into the
retrieval layer.
"""
from typing import List, Dict, Any

# Only these clinical fields are considered "retrieval-relevant". Anything
# else on the patient object (Age, Gender, Smoking history, etc.) is not
# used to build queries -- it doesn't map to a useful evidence lookup and
# keeping it out avoids leaking unnecessary patient detail into retrieval.
RETRIEVAL_RELEVANT_FEATURES = {
    "Response",
    "Risk",
    "Stage",
    "T",
    "N",
    "M",
    "Pathology",
    "Adenopathy",
    "Focality",
}

# Human-readable phrasing per feature, used to turn "Response=Excellent"
# into "excellent response differentiated thyroid cancer recurrence".
_FEATURE_PHRASING = {
    "Response": lambda v: f"{v.lower()} response differentiated thyroid cancer recurrence",
    "Risk": lambda v: f"{v.lower()} risk differentiated thyroid cancer recurrence",
    "Stage": lambda v: f"stage {v} differentiated thyroid cancer follow-up",
    "T": lambda v: f"{v} tumor classification thyroid cancer TNM staging",
    "N": lambda v: f"{v} nodal status thyroid cancer TNM staging",
    "M": lambda v: f"{v} metastasis thyroid cancer TNM staging",
    "Pathology": lambda v: f"{v.lower()} thyroid cancer pathology recurrence",
    "Adenopathy": lambda v: f"{v.lower()} adenopathy thyroid cancer recurrence risk",
    "Focality": lambda v: f"{v.lower()} thyroid cancer focality recurrence risk",
}


def build_queries(
    patient: Dict[str, Any],
    shap_factors: List[Dict[str, Any]],
    max_queries: int = 4,
) -> List[str]:
    """
    Build retrieval queries primarily from the top SHAP factors (the
    features the ML model itself flagged as most influential), falling
    back to a couple of core patient fields (Risk/Stage/Response) if SHAP
    factors are missing or none are retrieval-relevant.
    """
    queries: List[str] = []
    seen_features = set()

    for factor in shap_factors or []:
        feature = factor.get("feature")
        if feature not in RETRIEVAL_RELEVANT_FEATURES or feature in seen_features:
            continue

        value = factor.get("value")
        if value is None:
            value = patient.get(feature)
        if value is None:
            continue

        phrase_fn = _FEATURE_PHRASING.get(feature)
        if phrase_fn is None:
            continue

        queries.append(phrase_fn(str(value)))
        seen_features.add(feature)
        if len(queries) >= max_queries:
            break

    # Fallback: if SHAP gave us nothing usable, fall back to a few core
    # fields straight off the patient profile so retrieval still has
    # something reasonable to work with.
    if not queries:
        for feature in ("Response", "Risk", "Stage"):
            value = patient.get(feature)
            if value is None:
                continue
            phrase_fn = _FEATURE_PHRASING.get(feature)
            if phrase_fn:
                queries.append(phrase_fn(str(value)))

    return queries[:max_queries]

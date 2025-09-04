# predict_and_explain.py
# Loads saluSCORE_ped_pipeline.pkl, predicts probability,
# computes per-patient SHAP (averaged across bagged XGB),
# prints traditional scores (RACHS/ABC/STS) and saves a SHAP bar plot.

import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from salu_pipeline_components import (
    SurgeryDeriver, BinaryEncoder, IterativeImputerWrapper,
    RoundingTransformer, PreFittedScaler, ColumnOrderEnforcer
)

PIPELINE_PKL = "saluSCORE_ped_pipeline.pkl"


def _unwrap_for_shap(model):
    """
    Return a SHAP-compatible underlying estimator:
    - CalibratedClassifierCV -> first calibrated fold's .estimator
    - Pipeline -> last step's estimator
    - otherwise return as-is
    Recurses until it reaches a non-calibrated, non-pipeline estimator.
    """
    # Unwrap CalibratedClassifierCV
    if isinstance(model, CalibratedClassifierCV):
        # prefer the first calibrated classifier (binary case)
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            inner = model.calibrated_classifiers_[0]
            # _CalibratedClassifier exposes the original estimator on .estimator
            if hasattr(inner, "estimator") and inner.estimator is not None:
                return _unwrap_for_shap(inner.estimator)
        # fallback: some versions keep a .base_estimator on the CV object
        if hasattr(model, "base_estimator") and model.base_estimator is not None:
            return _unwrap_for_shap(model.base_estimator)
        raise RuntimeError("Could not unwrap CalibratedClassifierCV to a base estimator for SHAP.")

    # Unwrap Pipeline → take the final step
    if isinstance(model, Pipeline):
        if hasattr(model, "steps") and model.steps:
            return _unwrap_for_shap(model.steps[-1][1])

    # Otherwise assume it's already an estimator (e.g., BaggingClassifier, XGBClassifier)
    return model


def predict_proba_and_shap(row: pd.DataFrame, max_display=10):
    """
    row: single-row DataFrame with raw inputs including 'surgery' multiselect (list or string).
    Returns: proba (float), shap_top (DataFrame), traditional (Series), fig (matplotlib Figure)
    """
    import numpy as np
    import pandas as pd
    import shap
    import matplotlib.pyplot as plt
    from sklearn.ensemble import BaggingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    import joblib

    pipe = joblib.load("saluSCORE_ped_pipeline.pkl")

    # Unpack steps so we can capture de-normalized values for display
    derive   = pipe.named_steps["derive"]
    encode   = pipe.named_steps["encode"]
    impute   = pipe.named_steps["impute"]
    rounder  = pipe.named_steps["round"]
    order    = pipe.named_steps["order"]
    scaler   = pipe.named_steps["scale"]
    model    = pipe.named_steps["model"]   # may be CalibratedClassifierCV

    X = row.copy()
    Xd = derive.transform(X)
    Xe = encode.transform(Xd)
    Xi = impute.transform(Xe)
    Xr = rounder.transform(Xi)         # <- values in original units for display
    Xo = order.transform(Xr)
    Xs = scaler.transform(Xo)          # <- scaled values seen by the model

    # --- 1) Probability from the calibrated model (if calibrated) ---
    proba = float(model.predict_proba(Xs)[:, 1])

    # --- 2) Choose a SHAP-compatible tree model (pre-calibration) ---
    shap_model = _unwrap_for_shap(model)

    if isinstance(shap_model, BaggingClassifier):
        # (same averaging logic you already have)
        shap_values_per_estimator = []
        expected_vals = []
        for i, est in enumerate(shap_model.estimators_):
            feat_idx = getattr(shap_model, "estimators_features_", [None]*len(shap_model.estimators_))[i]
            X_subset = Xs if feat_idx is None else Xs.iloc[:, feat_idx]
            expl = shap.TreeExplainer(est)
            sv = expl.shap_values(X_subset)
            padded = np.zeros((1, Xs.shape[1]))
            if feat_idx is None:
                padded[:] = sv
            else:
                padded[:, feat_idx] = sv
            shap_values_per_estimator.append(padded)
            expected_vals.append(expl.expected_value)
        shap_values_avg = np.mean(shap_values_per_estimator, axis=0).flatten()
        base_value = float(np.mean(expected_vals))
    else:
        expl = shap.TreeExplainer(shap_model)
        sv = expl.shap_values(Xs)
        shap_values_avg = np.array(sv).reshape(-1)
        base_value = float(expl.expected_value)


    # --- 3) Compute SHAP on the underlying tree model ---
    # Handle Bagging of tree models (e.g., BaggingClassifier of XGBClassifiers)
    if isinstance(shap_model, BaggingClassifier):
        shap_values_per_estimator = []
        expected_vals = []
        for i, est in enumerate(shap_model.estimators_):
            # Subset to features used by this base estimator
            feat_idx = getattr(shap_model, "estimators_features_", [None]*len(shap_model.estimators_))[i]
            if feat_idx is None:
                X_subset = Xs
            else:
                X_subset = Xs.iloc[:, feat_idx]

            expl = shap.TreeExplainer(est)
            sv = expl.shap_values(X_subset)  # shape (1, n_used_features)
            # pad back to full feature space for averaging
            padded = np.zeros((1, Xs.shape[1]))
            if feat_idx is None:
                padded[:] = sv
            else:
                padded[:, feat_idx] = sv
            shap_values_per_estimator.append(padded)
            expected_vals.append(expl.expected_value)

        shap_values_avg = np.mean(shap_values_per_estimator, axis=0).flatten()
        base_value = float(np.mean(expected_vals))

    else:
        # Single tree model (e.g., xgboost.XGBClassifier)
        expl = shap.TreeExplainer(shap_model)
        sv = expl.shap_values(Xs)
        shap_values_avg = np.array(sv).reshape(-1)
        base_value = float(expl.expected_value)

    # --- 4) Build tidy SHAP frame using de-normalized display values ---
    feat_names = Xs.columns
    display_vals = Xr.iloc[0].reindex(feat_names)  # values in original units (post-rounding)
    shap_df = pd.DataFrame({
        "feature": feat_names,
        "value":   display_vals.values.astype(float),
        "shap":    shap_values_avg
    }).sort_values("shap", key=lambda s: s.abs(), ascending=False)
    shap_top = shap_df.head(max_display)

    # --- 5) In-memory SHAP bar figure (use JSON labels + mark imputed) ---
    import json

    # map internal feature names -> human labels from feature_schema.json
    with open("feature_schema.json", "r", encoding="utf-8") as f:
        schema_json = json.load(f)
    name_to_label = {fld["name"]: fld.get("label", fld["name"]) for fld in schema_json.get("fields", [])}

    # Build display names for the left y-axis: label, and note if imputed
    display_names = []
    for feat in shap_top["feature"].values:
        label = name_to_label.get(feat, feat)
        v = display_vals.get(feat, np.nan)  # value shown on the left ("v = label")
        if pd.isna(v):
            label = f"{label} (imputed)"
        display_names.append(label)

    explanation = shap.Explanation(
        values=shap_top["shap"].values,
        base_values=base_value,
        data=shap_top["value"].values,        # the numeric values shown before '='
        feature_names=np.array(display_names)  # human labels on the left
    )
    fig = plt.figure()
    shap.plots.bar(explanation, max_display=max_display, show=False)
    plt.title("Top Feature Contributions")


    # --- 6) Traditional derived scores from de-normalized frame ---
    trad_cols = ["rachs", "abc level", "abc score", "stmort category", "stmort score"]
    available = [c for c in trad_cols if c in Xr.columns]
    traditional = Xr.iloc[0][available]

    return proba, shap_top, traditional, fig



if __name__ == "__main__":
    # Example usage
    import pandas as pd
    example = pd.DataFrame([{
        "gender": "male", "age": 12.50, "weight": 32.1, "spo2": 98, "downs": "no",
        "hb": 13.1, "hct": 41.2, "mcv": 84.7, "mch": 29.2, "mchc": 34.5,
        "plt": 250, "tlc": 7.5, "inr": 1.1, "ptt": 32.4, "creat": 0.6, "urea": 20,
        "alt": 28, "ast": 30,
        "surgery": ["vsd patch", "pv valvuplasty"]
    }])

    proba, shap_top, traditional, fig = predict_proba_and_shap(example, max_display=10)
    print(f"Predicted mortality probability: {proba*100:.2f}%")
    print("\nTop contributors (SHAP):")
    print(shap_top.to_string(index=False))
    print("\nTraditional scores:")
    for k, v in traditional.items():
        print(f"  {k}: {v}")
    # Show the plot interactively if you run this file directly
    import matplotlib.pyplot as plt
    plt.show()

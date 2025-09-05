# predict_and_explain.py
# Loads saluSCORE_ped_pipeline.pkl, predicts probability,
# computes per-patient SHAP (averaged across bagged XGB),
# prints traditional scores (RACHS/ABC/STS) and saves a SHAP bar plot.

import joblib
import shap
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from salu_pipeline_components import (
    SurgeryDeriver, BinaryEncoder, IterativeImputerWrapper,
    RoundingTransformer, PreFittedScaler, ColumnOrderEnforcer
)
    

PIPELINE_PKL = "saluSCORE_ped_pipeline.pkl"

def get_imputed_raw_values(pipe: Pipeline, row: pd.DataFrame,
                           numeric_block_names=("num", "numeric", "numerical"),
                           imputer_step_name="imputer") -> dict:
    """
    Returns {feature_name: value_used_by_model_in_raw_units} for the given single-row DataFrame.
    If a value was missing in `row`, it is replaced by the number produced by the numeric imputer.
    Works for a Pipeline that contains a ColumnTransformer with a numeric pipeline
    that has an 'imputer' step (SimpleImputer/IterativeImputer).
    Falls back to the original row values if the structure isn't found.
    """
    # start with the original input values
    out = row.iloc[0].to_dict().copy()

    # find the ColumnTransformer inside the pipeline
    pre = None
    if isinstance(pipe, Pipeline):
        for _, step in pipe.named_steps.items():
            if isinstance(step, ColumnTransformer):
                pre = step
                break
            # some pipelines: a 'preprocess' step is itself a Pipeline
            if isinstance(step, Pipeline):
                for __, sub in step.named_steps.items():
                    if isinstance(sub, ColumnTransformer):
                        pre = sub
                        break
            if pre is not None:
                break
    elif isinstance(pipe, ColumnTransformer):
        pre = pipe

    if pre is None:
        return out  # can't locate preprocessing; return inputs as-is

    # find the numeric transformer + its columns
    num_name = None
    num_cols = None
    num_pipe = None

    # try fitted 'transformers_' first, then the declared 'transformers'
    for name, transformer, cols in getattr(pre, "transformers_", getattr(pre, "transformers", [])):
        if name in numeric_block_names:
            num_name = name
            num_cols = list(cols) if not isinstance(cols, slice) else list(row.columns[cols])
            num_pipe = transformer
            break

    if num_pipe is None or num_cols is None:
        return out

    # get the imputer step (pipeline or bare imputer)
    imputer = None
    if isinstance(num_pipe, Pipeline):
        if imputer_step_name in num_pipe.named_steps and isinstance(num_pipe.named_steps[imputer_step_name], (SimpleImputer, )):
            imputer = num_pipe.named_steps[imputer_step_name]
    elif isinstance(num_pipe, (SimpleImputer, )):
        imputer = num_pipe

    if imputer is None:
        # could be IterativeImputer etc.; try attribute presence instead of type
        if isinstance(num_pipe, Pipeline) and hasattr(num_pipe, "named_steps") and imputer_step_name in num_pipe.named_steps:
            imputer = num_pipe.named_steps[imputer_step_name]
        else:
            return out

    # run the imputer only on numeric columns in raw scale
    Xn = row[num_cols]
    try:
        Xn_imp = pd.DataFrame(imputer.transform(Xn), columns=num_cols, index=row.index)
    except Exception:
        # some imputers require fit; if not fitted, bail out gracefully
        return out

    for c in num_cols:
        if pd.isna(out.get(c)):
            out[c] = float(Xn_imp.loc[row.index[0], c])

    return out


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

    # --- 5) SHAP bar figure: single value; "(imputed)" drawn in grey next to the value ---
    # map internal names -> human labels
    with open("feature_schema.json", "r", encoding="utf-8") as f:
        schema_json = json.load(f)
    name_to_label = {fld["name"]: fld.get("label", fld["name"]) for fld in schema_json.get("fields", [])}

    # features (order as in shap_top)
    feats = list(shap_top["feature"].values)

    # values to show (prefer post-imputation+rounding from Xr)
    def pick_value(feat):
        # 1) value after pipeline imputer + rounding
        if feat in Xr.columns and pd.notna(Xr.iloc[0][feat]):
            return float(Xr.iloc[0][feat])
        # 2) fallback: imputer-only helper (raw)
        if "get_imputed_raw_values" in globals():
            v_used = get_imputed_raw_values(pipe, row).get(feat, np.nan)
            if pd.notna(v_used):
                return float(v_used)
        # 3) fallback: value column carried with shap_top
        try:
            v = shap_top.loc[shap_top["feature"] == feat, "value"].values[0]
            return float(v) if pd.notna(v) else np.nan
        except Exception:
            return np.nan

    data_values = [pick_value(f) for f in feats]
    feature_labels = [name_to_label.get(f, f) for f in feats]
    # original user inputs – to decide if "(imputed)" tag is needed
    imputed_flags = [pd.isna(row.iloc[0].get(f, np.nan)) for f in feats]

    # build explanation and plot
    explanation = shap.Explanation(
        values=shap_top["shap"].values,
        base_values=base_value,
        data=np.array(data_values),
        feature_names=np.array(feature_labels)
    )
    fig = plt.figure()
    shap.plots.bar(explanation, max_display=max_display, show=False)
    plt.title("Top Factors Influencing the Risk Estimate")

    # add a grey "(imputed)" *near the number* (not in the label to avoid overlap)
    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    # this offset controls where the grey tag appears relative to the left edge;
    # increase if it sits too close to the plot border; decrease if it bumps into labels
    x_hint = xmin + 0.12 * (xmax - xmin)

    for y, was_imputed in zip(ax.get_yticks(), imputed_flags):
        if was_imputed:
            ax.text(
                x_hint, y, "(imputed)",
                va="center", ha="left",
                color="#6B7280", fontsize=plt.rcParams['font.size'] * 0.9
            )


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

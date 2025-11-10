# build_serving_pipeline.py
# Creates ONE serving artifact: saluSCORE_ped_pipeline.pkl
# Steps: derive → encode → impute → round → order → scale → model

import json
import joblib
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import List, Dict, Any, Sequence, Optional

# sklearn
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from salu_pipeline_components import (
    SurgeryDeriver, BinaryEncoder, IterativeImputerWrapper,
    RoundingTransformer, PreFittedScaler, ColumnOrderEnforcer
)

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- Preflight: ensure stdlib pathlib is used (not a backport or local package) ---
import pathlib, sys
_pf = getattr(pathlib, "__file__", "") or ""
if "site-packages" in _pf.lower():
    raise RuntimeError(f"Bad pathlib found at: {_pf}. Remove any 'pathlib' wheels/backports or local modules.")

# --------------------
# Paths (adjust if needed)
# --------------------
FEATURE_SCHEMA = Path("feature_schema.json")                  # your schema JSON
LOOKUP_CSV   = Path("complexities_imputed.csv")            # surgeries → scores
TRAIN_CSV     = Path("training_data_fitting_imputer.csv")    # for fitting imputer
SCALER_JOBLIB  = Path("normalization_scaler.joblib")          # pre-fitted MinMaxScaler
MODEL_PKL      = Path("calibrated_xgboost_bagging_model.pkl") # trained + calibrated model
OUT_PIPELINE   = Path("saluSCORE_ped_pipeline.pkl")

TARGET_COL     = "mort in"


# --------------------
# Utilities
# --------------------
def read_schema(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------
# Build the pipeline
# --------------------
def main():
    schema = read_schema(FEATURE_SCHEMA)
    feature_order = schema["feature_order"]

    # Rounding rules (from your training)
    rounding = {
        "ones":       ["gender", "spo2", "downs", "plt", "urea", "alt", "ast"],
        "tenths":     ["age", "weight", "hb", "hct", "mcv", "mch", "mchc", "ptt"],
        "hundredths": ["tlc", "inr", "creat"]
    }

    # 1) Fit IterativeImputer(RF) on TRAIN (exclude target)
    train = pd.read_csv(TRAIN_CSV)
    train_no_target = train.drop(columns=[TARGET_COL]) if TARGET_COL in train.columns else train

    # numeric columns used for imputation
    num_cols = train_no_target.select_dtypes(include=["float64","int64","float32","int32"]).columns.tolist()
    impute_cols = [c for c in num_cols if train_no_target[c].isnull().any()]

    imputer_wrapped = IterativeImputerWrapper(impute_cols=impute_cols).fit(train_no_target)

    # 2) Load pre-fitted scaler
    scaler = joblib.load(SCALER_JOBLIB)

    # 3) Load trained model (already calibrated+bagged)
    try:
        model = joblib.load(MODEL_PKL)
    except Exception as e:
        warnings.warn(
            "Failed to load model pickle. This usually indicates a Python/numpy/"
            "scikit-learn/xgboost version mismatch. Use the same versions as training.\n"
            f"Error: {e}"
        )
        raise

    # 4) Build the serving pipeline
    pipe = Pipeline(steps=[
        ("derive", SurgeryDeriver(lookup_path=LOOKUP_CSV)),
        ("encode", BinaryEncoder(mappings={
            "gender": {"male": 1, "female": 0},
            "downs":  {"yes": 1, "no": 0}
        })),
        ("impute", imputer_wrapped),
        ("round",  RoundingTransformer(**rounding)),
        ("order",  ColumnOrderEnforcer(feature_order)),
        ("scale",  PreFittedScaler(scaler)),
        ("model",  model)
    ])

    # a) Fit SurgeryDeriver so its lookup table (_map) is initialized
    derive = pipe.named_steps["derive"]
    derive.fit(pd.DataFrame({"surgery": []}))  # fit reads CSV; X content not used

    # b) Build a representative frame through the earlier steps so scaler can learn its column list
    Xtmp = pd.read_csv(TRAIN_CSV)
    if TARGET_COL in Xtmp.columns:
        Xtmp = Xtmp.drop(columns=[TARGET_COL])

    Xtmp = derive.transform(Xtmp)
    Xtmp = pipe.named_steps["encode"].transform(Xtmp)
    Xtmp = pipe.named_steps["impute"].transform(Xtmp)
    Xtmp = pipe.named_steps["round"].transform(Xtmp)
    Xtmp = pipe.named_steps["order"].transform(Xtmp)

    # c) Fit the pre-fitted scaler wrapper (it doesn't refit MinMaxScaler; it just records which columns to transform)
    pipe.named_steps["scale"].fit(Xtmp)

    joblib.dump(pipe, OUT_PIPELINE, compress=3)
    print(f"Saved serving pipeline → {OUT_PIPELINE.resolve()}")

if __name__ == "__main__":
    main()

# build_serving_pipeline.py
# Builds serving artifact saluSCORE_ped_pipeline.pkl.
# If missing, also builds:
#   - normalization_scaler.joblib  (from B_Train80_NoLKG_Imputed.xlsx)
#   - calibrated_xgboost_bagging_model.pkl (from B_Train80_NoLKG_Imputed_NORM_RmvZero.xlsx)

import json
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# sklearn
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_curve
from sklearn.preprocessing import MinMaxScaler

# xgboost
import xgboost as xgb

# local
from salu_pipeline_components import (
    SurgeryDeriver, BinaryEncoder, IterativeImputerWrapper,
    RoundingTransformer, PreFittedScaler, ColumnOrderEnforcer
)

warnings.filterwarnings("ignore")

# --------------------
# Paths
# --------------------
FEATURE_SCHEMA = Path("feature_schema.json")
LOOKUP_CSV = Path("complexities_imputed.csv")
TRAIN_CSV_FOR_IMPUTER = Path("training_data_fitting_imputer.csv")  # used to fit IterativeImputer
SCALER_JOBLIB = Path("normalization_scaler.joblib")
MODEL_PKL = Path("calibrated_xgboost_bagging_model.pkl")
OUT_PIPELINE = Path("saluSCORE_ped_pipeline.pkl")

# your original training inputs
TRAIN_XLSX_FOR_SCALER = Path("B_Train80_NoLKG_Imputed.xlsx")
TRAIN_XLSX_FOR_MODEL = Path("B_Train80_NoLKG_Imputed_NORM_RmvZero.xlsx")
TARGET_COL = "mort in"

# --------------------
# Helpers
# --------------------
def read_schema(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_scaler():
    if SCALER_JOBLIB.exists():
        print(f"[scaler] Found existing: {SCALER_JOBLIB}")
        return

    print(f"[scaler] Building scaler from: {TRAIN_XLSX_FOR_SCALER}")
    df_train = pd.read_excel(TRAIN_XLSX_FOR_SCALER)
    numerical_cols = df_train.select_dtypes(include=["float64", "int64"]).columns
    scaler = MinMaxScaler()
    scaler.fit(df_train[numerical_cols])
    joblib.dump(scaler, SCALER_JOBLIB)
    print(f"[scaler] Saved: {SCALER_JOBLIB}")

def create_xgb_model():
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        max_depth=3,
        learning_rate=0.01928044904842136,
        min_child_weight=1,
        subsample=0.8614501903065227,
        colsample_bytree=0.8796696936441701,
        colsample_bylevel=0.7324794662581371,
        gamma=1.193696245915529,
        n_estimators=100,
        scale_pos_weight=5,
        reg_alpha=0.00032536986825252743,
        reg_lambda=0.5447906698870625
    )

def ensure_model():
    if MODEL_PKL.exists():
        print(f"[model] Found existing: {MODEL_PKL}")
        return

    print(f"[model] Training calibrated bagged XGBoost from: {TRAIN_XLSX_FOR_MODEL}")
    train_df = pd.read_excel(TRAIN_XLSX_FOR_MODEL)
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    bagging = BaggingClassifier(
        estimator=create_xgb_model(),
        n_estimators=12,
        max_samples=0.9106169555339139,
        max_features=0.862727747704497,
        bootstrap=True,
        random_state=42
    )

    calibrated = CalibratedClassifierCV(bagging, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)

    joblib.dump(calibrated, MODEL_PKL)
    print(f"[model] Saved: {MODEL_PKL}")

def build_serving_pipeline():
    schema = read_schema(FEATURE_SCHEMA)
    feature_order = schema["feature_order"]

    rounding = {
        "ones":       ["gender", "spo2", "downs", "plt", "urea", "alt", "ast"],
        "tenths":     ["age", "weight", "hb", "hct", "mcv", "mch", "mchc", "ptt"],
        "hundredths": ["tlc", "inr", "creat"]
    }

    train = pd.read_csv(TRAIN_CSV_FOR_IMPUTER)
    train_no_target = train.drop(columns=[TARGET_COL]) if TARGET_COL in train.columns else train

    num_cols = train_no_target.select_dtypes(include=["float64","int64","float32","int32"]).columns
    impute_cols = [c for c in num_cols if train_no_target[c].isnull().any()]

    imputer_wrapped = IterativeImputerWrapper(impute_cols=impute_cols).fit(train_no_target)

    scaler = joblib.load(SCALER_JOBLIB)
    model = joblib.load(MODEL_PKL)

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
        ("model",  model),
    ])

    pipe.named_steps["derive"].fit(pd.DataFrame({"surgery": []}))

    Xtmp = train_no_target.copy()
    Xtmp = pipe.named_steps["derive"].transform(Xtmp)
    Xtmp = pipe.named_steps["encode"].transform(Xtmp)
    Xtmp = pipe.named_steps["impute"].transform(Xtmp)
    Xtmp = pipe.named_steps["round"].transform(Xtmp)
    Xtmp = pipe.named_steps["order"].transform(Xtmp)

    pipe.named_steps["scale"].fit(Xtmp)

    joblib.dump(pipe, OUT_PIPELINE, compress=3)
    print(f"[pipeline] Saved → {OUT_PIPELINE.resolve()}")

def main():
    ensure_scaler()
    ensure_model()
    build_serving_pipeline()

if __name__ == "__main__":
    main()

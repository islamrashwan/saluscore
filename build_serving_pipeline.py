# build_serving_pipeline.py
# Assembles the final serving pipeline: saluSCORE_ped_pipeline.pkl

import json
import joblib
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from salu_pipeline_components import (
    SurgeryDeriver, BinaryEncoder, IterativeImputerWrapper,
    RoundingTransformer, PreFittedScaler, ColumnOrderEnforcer
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# -----------------------------------------------------------
# File paths (these will exist inside Docker after training)
# -----------------------------------------------------------
FEATURE_SCHEMA   = Path("feature_schema.json")
LOOKUP_CSV       = Path("complexities_imputed.csv")
TRAIN_CSV        = Path("training_data_fitting_imputer.csv")

SCALER_FILE      = Path("normalization_scaler.joblib")               # created in Docker
MODEL_FILE       = Path("calibrated_xgboost_bagging_model.pkl")      # created in Docker

OUT_PIPELINE     = Path("saluSCORE_ped_pipeline.pkl")

TARGET_COL       = "mort in"


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------
def ensure_exists(path: Path, name: str):
    """Ensure required artifact exists. If not, give a clear error."""
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing required {name}: {path}\n"
            f"This file is generated during Docker build via train_model_and_scaler.py.\n"
            f"Make sure your Dockerfile includes:\n"
            f"  RUN python train_model_and_scaler.py\n"
        )


def read_schema(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------
# Main assembly
# -----------------------------------------------------------
def main():
    print("\n=== Building Serving Pipeline ===")

    # 1. Ensure necessary files exist
    ensure_exists(FEATURE_SCHEMA, "feature schema JSON")
    ensure_exists(LOOKUP_CSV, "surgery lookup CSV")
    ensure_exists(TRAIN_CSV, "training CSV for imputer")

    ensure_exists(SCALER_FILE, "normalization scaler")
    ensure_exists(MODEL_FILE, "calibrated model pickle")

    # 2. Load schema
    schema = read_schema(FEATURE_SCHEMA)
    feature_order = schema["feature_order"]

    # 3. Load scaler
    print("Loading scaler...")
    scaler = joblib.load(SCALER_FILE)

    # 4. Load calibrated model
    print("Loading calibrated model...")
    model = joblib.load(MODEL_FILE)

    # 5. Fit imputer using TRAIN_CSV
    print("Fitting imputer on training data...")
    train = pd.read_csv(TRAIN_CSV)
    train_no_target = train.drop(columns=[TARGET_COL]) if TARGET_COL in train.columns else train

    numeric_cols = train_no_target.select_dtypes(include=["float64","float32","int64","int32"]).columns
    impute_cols = [c for c in numeric_cols if train_no_target[c].isnull().any()]
    imputer = IterativeImputerWrapper(impute_cols=impute_cols).fit(train_no_target)

    # 6. Build pipeline
    print("Assembling pipeline...")

    rounding = {
        "ones":       ["gender", "spo2", "downs", "plt", "urea", "alt", "ast"],
        "tenths":     ["age", "weight", "hb", "hct", "mcv", "mch", "mchc", "ptt"],
        "hundredths": ["tlc", "inr", "creat"]
    }

    pipe = Pipeline([
        ("derive", SurgeryDeriver(lookup_path=LOOKUP_CSV)),
        ("encode", BinaryEncoder(mappings={
            "gender": {"male": 1, "female": 0},
            "downs":  {"yes": 1, "no": 0},
        })),
        ("impute", imputer),
        ("round",  RoundingTransformer(**rounding)),
        ("order",  ColumnOrderEnforcer(feature_order)),
        ("scale",  PreFittedScaler(scaler)),
        ("model",  model)
    ])

    # 7. Fit SurgeryDeriver (loads lookup table)
    print("Initializing SurgeryDeriver...")
    pipe.named_steps["derive"].fit(pd.DataFrame({"surgery": []}))

    # 8. Pass sample data through early steps to initialize scaling columns
    print("Preparing scale-column mapping...")
    X_tmp = pd.read_csv(TRAIN_CSV)
    X_tmp = X_tmp.drop(columns=[TARGET_COL]) if TARGET_COL in X_tmp else X_tmp

    X_tmp = pipe.named_steps["derive"].transform(X_tmp)
    X_tmp = pipe.named_steps["encode"].transform(X_tmp)
    X_tmp = pipe.named_steps["impute"].transform(X_tmp)
    X_tmp = pipe.named_steps["round"].transform(X_tmp)
    X_tmp = pipe.named_steps["order"].transform(X_tmp)

    pipe.named_steps["scale"].fit(X_tmp)

    # 9. Save final pipeline
    print(f"Saving final pipeline → {OUT_PIPELINE} ...")
    joblib.dump(pipe, OUT_PIPELINE, compress=3)

    print("✅ Pipeline build complete.\n")


if __name__ == "__main__":
    main()

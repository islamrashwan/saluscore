# train_model_and_scaler.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_curve
import xgboost as xgb
import joblib

TRAIN_XLSX_RAW = "training_assets/B_Train80_NoLKG_Imputed.xlsx"
TRAIN_XLSX_NORM_RMVZERO = "training_assets/B_Train80_NoLKG_Imputed_NORM_RmvZero.xlsx"

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


def train_scaler():
    print("Loading raw training data for scaler...")
    df = pd.read_excel(TRAIN_XLSX_RAW)

    numeric = df.select_dtypes(include=["int64","float64"]).columns
    scaler = MinMaxScaler()
    df[numeric] = scaler.fit_transform(df[numeric])

    print("Saving scaler...")
    joblib.dump(scaler, "normalization_scaler.joblib")

    print("Saving normalized dataset...")
    df.to_excel("training_assets/B_Train80_NoLKG_Imputed_NORM.xlsx", index=False)


def train_model():
    print("Loading normalized dataset (RmvZero) for model...")
    df = pd.read_excel(TRAIN_XLSX_NORM_RMVZERO)

    X = df.drop("mort in", axis=1)
    y = df["mort in"]

    bag = BaggingClassifier(
        estimator=create_xgb_model(),
        n_estimators=12,
        max_samples=0.9106,
        max_features=0.8627,
        bootstrap=True,
        random_state=42
    )

    print("Applying isotonic calibration with CV=5...")
    calibrated = CalibratedClassifierCV(bag, cv=5, method="isotonic")

    print("Getting cross-validated probabilities for threshold...")
    y_proba = cross_val_predict(calibrated, X, y, cv=5, method="predict_proba")[:, 1]

    fpr, tpr, thr = roc_curve(y, y_proba)
    youden = tpr + (1 - fpr) - 1
    best_thr = thr[np.argmax(youden)]

    print(f"Best Youden threshold: {best_thr}")

    print("Training final calibrated model on full dataset...")
    calibrated.fit(X, y)

    print("Saving calibrated model...")
    joblib.dump(calibrated, "calibrated_xgboost_bagging_model.pkl")


def main():
    print("=== Training scaler ===")
    train_scaler()
    print("=== Training model ===")
    train_model()
    print("All training artifacts saved.")


if __name__ == "__main__":
    main()

# salu_pipeline_components.py

from pathlib import Path
from typing import Sequence, Optional, Any, Dict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


class SurgeryDeriver(BaseEstimator, TransformerMixin):
    """
    Derive RACHS/ABC/STS and flags from a multiselect 'surgery' column using a lookup CSV.
    Aggregation = max for numeric scores; flags = 1 if present else 0.
    """
    def __init__(self, lookup_path: Path,
                 score_cols=("rachs","abc level","abc score","stmort category","stmort score"),
                 flag_surgeries=("cavc repair","rv infundibulectomy","tof rv to pa conduit")):
        self.lookup_path = Path(lookup_path)
        self.score_cols = list(score_cols)
        self.flag_surgeries = list(flag_surgeries)
        self._map: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y=None):
        df = pd.read_csv(self.lookup_path)
        need = ["surgery", *self.score_cols]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"Lookup CSV missing columns: {missing}")
        df["surgery_norm"] = df["surgery"].astype(str).str.strip().str.lower()
        self._map = df.set_index("surgery_norm")[list(self.score_cols)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        import re
        X = X.copy()
        re_split = re.compile(r"[;,|]+")

        def parse_sel(val):
            if isinstance(val, (list, tuple, set)):
                return [str(v).strip().lower() for v in val]
            if pd.isna(val):
                return []
            parts = [p.strip().lower() for p in re_split.split(str(val)) if p.strip()]
            return parts

        selected_lists = X["surgery"].apply(parse_sel) if "surgery" in X.columns else pd.Series([[]]*len(X), index=X.index)

        # Aggregate max for each score column
        for col in self.score_cols:
            agg_vals = []
            for sels in selected_lists:
                if not sels:
                    agg_vals.append(np.nan)
                else:
                    vals = []
                    for s in sels:
                        if s in self._map.index:
                            val = self._map.loc[s, col]
                            if pd.notna(val):
                                vals.append(val)
                    agg_vals.append(max(vals) if vals else np.nan)
            X[col] = agg_vals

        # Flags: 1 if present
        for flag in self.flag_surgeries:
            key = flag.strip().lower()
            X[flag] = selected_lists.apply(lambda sels: 1 if key in sels else 0)

        # Drop raw surgery column (model uses derived fields)
        if "surgery" in X.columns:
            X = X.drop(columns=["surgery"])
        return X


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Map categorical text to 0/1 as requested."""
    def __init__(self, mappings: Dict[str, Dict[Any, int]]):
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, m in self.mappings.items():
            if col in X.columns:
                X[col] = X[col].map(m).astype("float64")
        return X


class IterativeImputerWrapper(BaseEstimator, TransformerMixin):
    """Fit IterativeImputer(RF) on training-only numeric columns; reuse at inference."""
    def __init__(self, impute_cols: Sequence[str]):
        self.impute_cols = list(impute_cols)
        self.imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=100, random_state=42),
            max_iter=10, random_state=42
        )

    def fit(self, X: pd.DataFrame, y=None):
        cols = [c for c in self.impute_cols if c in X.columns]
        self.impute_cols_ = cols
        if cols:
            self.imputer.fit(X[cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cols = [c for c in getattr(self, "impute_cols_", []) if c in X.columns]
        if cols:
            X[cols] = self.imputer.transform(X[cols])
        return X


class RoundingTransformer(BaseEstimator, TransformerMixin):
    """Apply rounding as in training; default rounding for other numeric columns."""
    def __init__(self, ones: Sequence[str], tenths: Sequence[str],
                 hundredths: Sequence[str], default_decimals: int = 3, cast_int_ones: bool = False):
        self.ones = list(ones)
        self.tenths = list(tenths)
        self.hundredths = list(hundredths)
        self.default_decimals = default_decimals
        self.cast_int_ones = cast_int_ones

    def fit(self, X, y=None): return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Specific groups
        for col in self.ones:
            if col in X.columns:
                arr = np.round(X[col].astype(float), 0)
                X[col] = arr.astype(int) if self.cast_int_ones else arr.astype("float64")
        for col in self.tenths:
            if col in X.columns:
                X[col] = np.round(X[col].astype(float), 1)
        for col in self.hundredths:
            if col in X.columns:
                X[col] = np.round(X[col].astype(float), 2)
        # Default for other numeric cols
        handled = set(self.ones) | set(self.tenths) | set(self.hundredths)
        for col in X.select_dtypes(include=[np.number]).columns:
            if col not in handled:
                X[col] = np.round(X[col].astype(float), self.default_decimals)
        return X


class PreFittedScaler(BaseEstimator, TransformerMixin):
    """
    Apply a pre-fitted MinMaxScaler in a way that preserves training-time columns.
    - If the scaler was fit with named columns (feature_names_in_), we:
      * create an augmented frame X_aug with exactly those columns (missing -> 0),
      * apply scaler.transform(X_aug),
      * then copy back the scaled values for any columns present in X.
    - If feature names aren't available, we fall back to transforming only numeric cols present.
    """
    def __init__(self, scaler):
        self.scaler = scaler

    def fit(self, X, y=None):
        # No fitting of the scaler here (it's already fitted).
        # We just remember which columns we will attempt to scale later.
        # Prefer the scaler's training-time feature names if available.
        self.scaler_feature_names_ = getattr(self.scaler, "feature_names_in_", None)
        # Keep a record of numeric columns we typically see at inference (best-effort fallback)
        self.infer_numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Case A: scaler has feature name memory (best)
        if self.scaler_feature_names_ is not None:
            expected = list(self.scaler_feature_names_)
            # Build X_aug with exactly the training-time columns
            # Fill any missing columns with 0 (neutral for MinMax scaling range on [min,max]; the scaled value will be consistent)
            X_aug = pd.DataFrame(
                {col: (X[col] if col in X.columns else 0.0) for col in expected},
                index=X.index
            )
            # Ensure numeric dtype
            for c in expected:
                X_aug[c] = pd.to_numeric(X_aug[c], errors="coerce")

            scaled = self.scaler.transform(X_aug)  # ndarray
            scaled_df = pd.DataFrame(scaled, columns=expected, index=X.index)

            # Copy back scaled values for the intersection columns present in X
            common = [c for c in expected if c in X.columns]
            if common:
                X[common] = scaled_df[common]
            # Columns present in X but not in expected are left untouched (no scaling was done at train-time)
            return X

        # Case B: no feature names on scaler → transform whatever numeric columns we have
        numeric_cols = [c for c in self.infer_numeric_cols_ if c in X.columns]
        if numeric_cols:
            X[numeric_cols] = self.scaler.transform(X[numeric_cols])
        return X



class ColumnOrderEnforcer(BaseEstimator, TransformerMixin):
    """Ensure final column order matches training/model expectation; insert missing as NaN."""
    def __init__(self, feature_order: Sequence[str]):
        self.feature_order = list(feature_order)

    def fit(self, X, y=None): return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.feature_order:
            if col not in X.columns:
                X[col] = np.nan
        return X[self.feature_order]

"""Class-based utilities for preparing and fitting IPCA models."""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ipca import InstrumentedPCA


@dataclass
class IPCAMatrices:
    """Container for matrices required by InstrumentedPCA."""

    X: np.ndarray
    y: np.ndarray
    indices: np.ndarray
    char_cols: List[str]


class IPCAWorkflow:
    """Stepwise helper class for OpenAP -> IPCA workflows."""

    def __init__(
        self,
        id_col: str = "permno",
        time_col: str = "yyyymm",
        ret_col: str = "excess_ret",
    ) -> None:
        self.id_col = id_col
        self.time_col = time_col
        self.ret_col = ret_col

    @staticmethod
    def load_wrds_credentials(credentials_path: str | Path) -> Tuple[str, str]:
        """Load WRDS credentials from JSON or KEY=VALUE flat file.

        Supported file formats:
        1) JSON: {"WRDS_USERNAME": "...", "WRDS_PASSWORD": "..."}
        2) env-like:
           WRDS_USERNAME=...
           WRDS_PASSWORD=...
        """
        path = Path(credentials_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Credentials file not found: {path}")

        text = path.read_text(encoding="utf-8").strip()
        data: Dict[str, str] = {}

        if path.suffix.lower() == ".json":
            parsed = json.loads(text)
            data = {str(k): str(v) for k, v in parsed.items()}
        else:
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                data[key.strip()] = value.strip().strip('"').strip("'")

        user = data.get("WRDS_USERNAME") or data.get("username")
        pwd = data.get("WRDS_PASSWORD") or data.get("password")
        if not user or not pwd:
            raise ValueError(
                "Credentials file must contain WRDS_USERNAME and WRDS_PASSWORD."
            )
        return user, pwd

    def download_openap_data(
        self,
        start_yyyymm: str,
        end_yyyymm: str,
        backend: str = "pandas",
    ) -> pd.DataFrame:
        """Download OpenAP signals and keep the requested date window."""
        try:
            import openassetpricing as oap
        except ImportError as exc:
            raise ImportError(
                "openassetpricing is required. Install with: pip install openassetpricing"
            ) from exc

        start_dt = pd.to_datetime(start_yyyymm, format="%Y%m")
        end_dt = pd.to_datetime(end_yyyymm, format="%Y%m")

        openap = oap.OpenAP()
        df = openap.dl_all_signals(backend)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        df[self.time_col] = pd.to_datetime(
            df[self.time_col].astype(str), format="%Y%m", errors="coerce"
        )
        df = df.dropna(subset=[self.id_col, self.time_col])
        df = df[(df[self.time_col] >= start_dt) & (df[self.time_col] <= end_dt)].copy()
        return df.sort_values([self.id_col, self.time_col]).reset_index(drop=True)

    def download_crsp_returns_wrds(
        self,
        start_date: str = "1980-01-01",
        end_date: Optional[str] = None,
        permnos: Optional[List[int]] = None,
        credentials_path: Optional[str | Path] = None,
        include_delist_adjustment: bool = True,
    ) -> pd.DataFrame:
        try:
            import wrds
        except ImportError as exc:
            raise ImportError("wrds package is required. Install with: pip install wrds") from exc

        if credentials_path:
            wrds_user, wrds_pass = self.load_wrds_credentials(credentials_path)
            db = wrds.Connection(wrds_username=wrds_user, wrds_password=wrds_pass)
        else:
            db = wrds.Connection()

        where_parts = ["msf.date >= %(start_date)s"]
        params: Dict[str, object] = {"start_date": start_date}
        if end_date is not None:
            where_parts.append("msf.date <= %(end_date)s")
            params["end_date"] = end_date
        if permnos:
            where_parts.append("msf.permno = ANY(%(permnos)s)")
            params["permnos"] = [int(p) for p in permnos]

        where_sql = " AND ".join(where_parts)

        if include_delist_adjustment:
            sql = f"""
                SELECT
                    msf.permno,
                    msf.date,
                    msf.ret,
                    dl.dlret,
                    msf.prc,
                    msf.shrout
                FROM crsp.msf AS msf
                LEFT JOIN crsp.msedelist AS dl
                    ON msf.permno = dl.permno
                   AND msf.date = dl.dlstdt
                WHERE {where_sql}
                ORDER BY msf.permno, msf.date
            """
        else:
            sql = f"""
                SELECT
                    msf.permno,
                    msf.date,
                    msf.ret,
                    msf.prc,
                    msf.shrout
                FROM crsp.msf AS msf
                WHERE {where_sql}
                ORDER BY msf.permno, msf.date
            """

        try:
            out = db.raw_sql(sql, params=params, date_cols=["date"])
        finally:
            db.close()

        out = out.copy()
        out["ret"] = pd.to_numeric(out["ret"], errors="coerce")
        if "dlret" in out.columns:
            out["dlret"] = pd.to_numeric(out["dlret"], errors="coerce")
            out["ret_adj"] = (1.0 + out["ret"].fillna(0.0)) * (
                1.0 + out["dlret"].fillna(0.0)
            ) - 1.0
            both_missing = out["ret"].isna() & out["dlret"].isna()
            out.loc[both_missing, "ret_adj"] = np.nan

        out[self.time_col] = pd.to_datetime(out["date"]).dt.to_period("M").dt.to_timestamp()
        return out

    def remove_mostly_nan_columns(
        self,
        df: pd.DataFrame,
        max_nan_frac: float = 0.6,
        protected_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Drop columns with NaN share above max_nan_frac."""
        if not 0.0 <= max_nan_frac <= 1.0:
            raise ValueError("max_nan_frac must be between 0 and 1.")

        if protected_cols is None:
            protected_cols = [self.id_col, self.time_col, "ret", "excess_ret", "year"]

        keep_cols: List[str] = []
        dropped_cols: List[str] = []

        for col in df.columns:
            if col in protected_cols:
                keep_cols.append(col)
                continue
            nan_frac = float(df[col].isna().mean())
            if nan_frac <= max_nan_frac:
                keep_cols.append(col)
            else:
                dropped_cols.append(col)

        return df[keep_cols].copy(), dropped_cols

    def merge_openap_with_crsp_returns(
        self,
        openap_df: pd.DataFrame,
        crsp_df: pd.DataFrame,
        crsp_return_col: str = "ret_adj",
        output_return_col: Optional[str] = None,
        how: str = "inner",
        keep_crsp_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Merge OpenAP characteristics with CRSP returns on entity-month.

        Parameters
        ----------
        openap_df : pd.DataFrame
            OpenAP-style panel with ``permno`` and ``yyyymm`` columns.
        crsp_df : pd.DataFrame
            Output from ``download_crsp_returns_wrds``.
        crsp_return_col : str
            CRSP return column to map into the merged panel (e.g., ``ret_adj`` or ``ret``).
        output_return_col : str, optional
            Name for return column in merged output. Defaults to ``self.ret_col``.
        how : str
            Merge mode, typically ``inner`` or ``left``.
        keep_crsp_columns : list[str], optional
            Extra CRSP columns to keep in output (e.g., ``["prc", "shrout"]``).
        """
        if output_return_col is None:
            output_return_col = self.ret_col

        if crsp_return_col not in crsp_df.columns:
            raise ValueError(
                f"CRSP return column '{crsp_return_col}' not found. Available: {list(crsp_df.columns)}"
            )

        left = openap_df.copy()
        right = crsp_df.copy()

        left[self.id_col] = pd.to_numeric(left[self.id_col], errors="coerce")
        right[self.id_col] = pd.to_numeric(right[self.id_col], errors="coerce")

        # Normalize both to month-start timestamps to guarantee matching keys.
        left[self.time_col] = pd.to_datetime(left[self.time_col], errors="coerce").dt.to_period(
            "M"
        ).dt.to_timestamp()

        if self.time_col in right.columns:
            right[self.time_col] = pd.to_datetime(right[self.time_col], errors="coerce").dt.to_period(
                "M"
            ).dt.to_timestamp()
        elif "date" in right.columns:
            right[self.time_col] = pd.to_datetime(right["date"], errors="coerce").dt.to_period(
                "M"
            ).dt.to_timestamp()
        else:
            raise ValueError(f"CRSP data must include '{self.time_col}' or 'date' column.")

        crsp_keep = [self.id_col, self.time_col, crsp_return_col]
        if keep_crsp_columns:
            for col in keep_crsp_columns:
                if col in right.columns and col not in crsp_keep:
                    crsp_keep.append(col)

        right = right[crsp_keep].rename(columns={crsp_return_col: output_return_col})
        merged = left.merge(right, on=[self.id_col, self.time_col], how=how, validate="m:1")

        if output_return_col in merged.columns:
            merged[output_return_col] = pd.to_numeric(merged[output_return_col], errors="coerce")

        return merged.sort_values([self.id_col, self.time_col]).reset_index(drop=True)

    def fill_remaining_missing(
        self,
        df: pd.DataFrame,
        protected_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fill remaining missing numeric values by month median then global median."""
        if protected_cols is None:
            protected_cols = [self.id_col, self.time_col, "ret", "excess_ret", "year"]

        out = df.copy()
        num_cols = [
            c for c in out.columns if c not in protected_cols and pd.api.types.is_numeric_dtype(out[c])
        ]

        out[num_cols] = out.groupby(self.time_col)[num_cols].transform(
            lambda s: s.fillna(s.median())
        )
        for col in num_cols:
            out[col] = out[col].fillna(out[col].median())

        return out

    def build_model_panel(
        self,
        df: pd.DataFrame,
        shift_target: bool = True,
        target_col: str = "y_ipca",
        extra_exclude: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Build model-ready panel with target and integer ids."""
        out = df.copy()
        out[self.time_col] = pd.to_datetime(
            out[self.time_col].astype(str), format="%Y%m", errors="coerce"
        )
        out = out.dropna(subset=[self.id_col, self.time_col]).sort_values(
            [self.id_col, self.time_col]
        )

        if shift_target:
            out[target_col] = out.groupby(self.id_col)[self.ret_col].shift(-1)
        else:
            out[target_col] = out[self.ret_col]

        exclude = {self.id_col, self.time_col, "ret", "excess_ret", "year", target_col}
        if extra_exclude:
            exclude.update(extra_exclude)
        char_cols = [
            c for c in out.columns if c not in exclude and pd.api.types.is_numeric_dtype(out[c])
        ]

        out = out[[self.id_col, self.time_col, target_col] + char_cols].replace(
            [np.inf, -np.inf], np.nan
        )
        out = out.dropna(subset=[target_col]).copy()

        out["i_idx"] = out[self.id_col].factorize(sort=True)[0].astype(np.int64)
        out["t_idx"] = out[self.time_col].factorize(sort=True)[0].astype(np.int64)
        return out

    def split_train_test(
        self,
        panel_df: pd.DataFrame,
        cutoff: str | pd.Timestamp = "2015-12-31",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split panel into train and test by calendar date."""
        cutoff_ts = pd.Timestamp(cutoff)
        train_df = panel_df[panel_df[self.time_col] <= cutoff_ts].copy()
        test_df = panel_df[panel_df[self.time_col] > cutoff_ts].copy()
        return train_df, test_df

    @staticmethod
    def normalize_train_test(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        char_cols: List[str],
        clip_quantiles: Optional[Tuple[float, float]] = (1.0, 99.0),
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Normalize using train-only statistics to avoid leakage."""
        train_out = train_df.copy()
        test_out = test_df.copy()

        mu = train_out[char_cols].mean(axis=0)
        sd = train_out[char_cols].std(axis=0, ddof=0).replace(0, 1.0)
        sd = sd.fillna(1.0)

        train_out[char_cols] = (train_out[char_cols] - mu) / sd
        test_out[char_cols] = (test_out[char_cols] - mu) / sd

        if clip_quantiles is not None:
            q_lo, q_hi = clip_quantiles
            lo = train_out[char_cols].quantile(q_lo / 100.0)
            hi = train_out[char_cols].quantile(q_hi / 100.0)
            train_out[char_cols] = train_out[char_cols].clip(lo, hi, axis=1)
            test_out[char_cols] = test_out[char_cols].clip(lo, hi, axis=1)

        return train_out, test_out

    @staticmethod
    def to_ipca_matrices(
        panel_df: pd.DataFrame,
        char_cols: List[str],
        target_col: str = "y_ipca",
    ) -> IPCAMatrices:
        """Convert panel DataFrame into X, y, indices."""
        X = panel_df[char_cols].to_numpy(np.float64)
        y = panel_df[target_col].to_numpy(np.float64)
        indices = panel_df[["i_idx", "t_idx"]].to_numpy(np.int64)

        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]
        indices = indices[mask]

        return IPCAMatrices(X=X, y=y, indices=indices, char_cols=char_cols)

    @staticmethod
    def fit_ipca(
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        n_factors: int = 3,
        intercept: bool = False,
        max_iter: int = 5000,
        iter_tol: float = 1e-4,
        silent: bool = True,
    ) -> InstrumentedPCA:
        """Fit IPCA model in panel mode."""
        model = InstrumentedPCA(
            n_factors=n_factors,
            intercept=intercept,
            max_iter=max_iter,
            iter_tol=iter_tol,
        )
        if silent:
            with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(
                fnull
            ), contextlib.redirect_stderr(fnull):
                model.fit(X, y, indices=indices, data_type="panel")
        else:
            model.fit(X, y, indices=indices, data_type="panel")
        return model

    @staticmethod
    def score_ipca(
        model: InstrumentedPCA,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        mean_factor: bool = False,
    ) -> float:
        """Compute model R^2 score in panel mode."""
        return float(
            model.score(
                X,
                y,
                indices=indices,
                data_type="panel",
                mean_factor=mean_factor,
            )
        )

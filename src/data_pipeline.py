"""Data loading, cleaning, and preparation utilities for IPCA workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class IPCAMatrices:
    """Container for matrices required by InstrumentedPCA."""

    X: np.ndarray
    y: np.ndarray
    indices: np.ndarray
    char_cols: List[str]


class DataPipeline:
    """Download, clean, and prepare characteristic panel data for IPCA."""

    def __init__(
        self,
        id_col: str = "permno",
        time_col: str = "yyyymm",
        ret_col: str = "excess_ret",
    ) -> None:
        self.id_col = id_col
        self.time_col = time_col
        self.ret_col = ret_col

    # ------------------------------------------------------------------
    # Data download
    # ------------------------------------------------------------------

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

        openap = oap.OpenAP()
        df = openap.dl_all_signals(backend)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        df[self.time_col] = pd.to_datetime(
            df[self.time_col].astype(str), format="%Y%m", errors="coerce"
        )
        df = df.dropna(subset=[self.id_col, self.time_col])
        return df.sort_values([self.id_col, self.time_col]).reset_index(drop=True)

    def download_sp500_returns_wrds(
        self,
        start_date: str = "1980-01-01",
        end_date: Optional[str] = None,
        include_delist_adjustment: bool = True,
    ) -> pd.DataFrame:
        """Download monthly returns for S&P 500 constituents only, point-in-time.

        Joins ``crsp.msf`` with ``crsp.msp500list`` in SQL so only stocks that
        were *actually in the index* at each observation date are returned —
        no survivorship bias, no post-hoc filtering required.

        Parameters
        ----------
        start_date : str
            First month to include (``YYYY-MM-DD``).
        end_date : str, optional
            Last month to include. Defaults to today.
        include_delist_adjustment : bool
            If True, compound raw return with delisting return (``dlret``) from
            ``crsp.msedelist`` and expose as ``ret_adj``.
        """
        try:
            import wrds
        except ImportError as exc:
            raise ImportError("wrds package is required. Install with: pip install wrds") from exc

        db = wrds.Connection()

        where_parts = ["msf.date >= %(start_date)s"]
        params: dict[str, object] = {"start_date": start_date}
        if end_date is not None:
            where_parts.append("msf.date <= %(end_date)s")
            params["end_date"] = end_date

        where_sql = " AND ".join(where_parts)

        # The INNER JOIN on msp500list enforces point-in-time S&P 500 membership:
        # a row is included only when msf.date falls within the stock's index spell.
        if include_delist_adjustment:
            sql = f"""
                SELECT
                    msf.permno,
                    msf.date,
                    msf.ret,
                    dl.dlret,
                    msf.prc,
                    msf.shrout,
                    msf.cfacpr,
                    msf.cfacshr
                FROM crsp.msf AS msf
                INNER JOIN crsp.msp500list AS sp
                    ON  msf.permno = sp.permno
                    AND msf.date  >= sp.start
                    AND (msf.date <= sp.ending OR sp.ending IS NULL)
                LEFT JOIN crsp.msedelist AS dl
                    ON  msf.permno = dl.permno
                    AND msf.date   = dl.dlstdt
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
                    msf.shrout,
                    msf.cfacpr,
                    msf.cfacshr
                FROM crsp.msf AS msf
                INNER JOIN crsp.msp500list AS sp
                    ON  msf.permno = sp.permno
                    AND msf.date  >= sp.start
                    AND (msf.date <= sp.ending OR sp.ending IS NULL)
                WHERE {where_sql}
                ORDER BY msf.permno, msf.date
            """

        try:
            out = db.raw_sql(sql, params=params, date_cols=["date"])
        finally:
            db.close()

        out = out.copy()
        out["ret"] = pd.to_numeric(out["ret"], errors="coerce")
        for col in ["prc", "shrout", "cfacpr", "cfacshr"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        if "dlret" in out.columns:
            out["dlret"] = pd.to_numeric(out["dlret"], errors="coerce")
            out["ret_adj"] = (1.0 + out["ret"].fillna(0.0)) * (
                1.0 + out["dlret"].fillna(0.0)
            ) - 1.0
            both_missing = out["ret"].isna() & out["dlret"].isna()
            out.loc[both_missing, "ret_adj"] = np.nan

        out[self.time_col] = pd.to_datetime(out["date"]).dt.to_period("M").dt.to_timestamp()
        return out

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

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
            protected_cols = [
                self.id_col,
                self.time_col,
                "ret",
                "ret_adj",
                "excess_ret",
                "year",
                "prc",
                "shrout",
                "cfacpr",
                "cfacshr",
                "mcap",
            ]

        protected_set = set(protected_cols)
        nan_fracs = df.isna().mean()
        dropped_cols = [
            c for c in df.columns
            if c not in protected_set and nan_fracs[c] > max_nan_frac
        ]
        keep_cols = [c for c in df.columns if c not in set(dropped_cols)]
        return df[keep_cols].copy(), dropped_cols

    def merge_openap_with_crsp_returns(
        self,
        openap_df: pd.DataFrame,
        crsp_df: pd.DataFrame,
        crsp_return_col: str = "ret_adj",
        output_return_col: Optional[str] = None,
        how: str = "inner",
        keep_crsp_columns: Optional[List[str]] = None,
        add_market_cap: bool = True,
    ) -> pd.DataFrame:
        """Merge OpenAP characteristics with CRSP returns on entity-month.

        Parameters
        ----------
        openap_df : pd.DataFrame
            OpenAP-style panel with ``permno`` and ``yyyymm`` columns.
        crsp_df : pd.DataFrame
            Output from ``download_sp500_returns_wrds``.
        crsp_return_col : str
            CRSP return column to map into the merged panel (e.g., ``ret_adj`` or ``ret``).
        output_return_col : str, optional
            Name for return column in merged output. Defaults to ``self.ret_col``.
        how : str
            Merge mode, typically ``inner`` or ``left``.
        keep_crsp_columns : list[str], optional
            Extra CRSP columns to keep in output. If omitted, standard CRSP
            price/share fields are retained when available:
            ``["prc", "shrout", "cfacpr", "cfacshr"]``.
        add_market_cap : bool
            If True and ``prc``/``shrout`` are present, add raw CRSP market cap
            as ``mcap = abs(prc) * shrout``.
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
        if keep_crsp_columns is None:
            keep_crsp_columns = ["prc", "shrout", "cfacpr", "cfacshr"]
        for col in keep_crsp_columns:
            if col in right.columns and col not in crsp_keep:
                crsp_keep.append(col)

        replace_cols = [output_return_col] + [c for c in keep_crsp_columns if c in left.columns]
        if add_market_cap and "mcap" in left.columns:
            replace_cols.append("mcap")
        replace_cols = [c for c in replace_cols if c in left.columns]
        if replace_cols:
            left = left.drop(columns=replace_cols)

        right = right[crsp_keep].rename(columns={crsp_return_col: output_return_col})
        merged = left.merge(right, on=[self.id_col, self.time_col], how=how, validate="m:1")

        if output_return_col in merged.columns:
            merged[output_return_col] = pd.to_numeric(merged[output_return_col], errors="coerce")
        for col in ["prc", "shrout", "cfacpr", "cfacshr"]:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")
        if add_market_cap and {"prc", "shrout"}.issubset(merged.columns):
            merged["mcap"] = merged["prc"].abs() * merged["shrout"]

        return merged.sort_values([self.id_col, self.time_col]).reset_index(drop=True)

    def fill_remaining_missing(
        self,
        df: pd.DataFrame,
        protected_cols: Optional[List[str]] = None,
        use_past_only: bool = False,
    ) -> pd.DataFrame:
        """Fill remaining numeric NaNs by same-month median, then fallback median.

        If ``use_past_only`` is False (default), the fallback is the full-sample
        global median for each column.

        If ``use_past_only`` is True, the fallback is a time-safe historical
        median built only from strictly earlier months.
        """
        if protected_cols is None:
            protected_cols = [
                self.id_col,
                self.time_col,
                "ret",
                "ret_adj",
                "excess_ret",
                "year",
                "prc",
                "shrout",
                "cfacpr",
                "cfacshr",
                "mcap",
            ]

        out = df.copy()
        num_cols = [
            c for c in out.columns if c not in protected_cols and pd.api.types.is_numeric_dtype(out[c])
        ]
        if not num_cols:
            return out

        if not use_past_only:
            global_medians = out[num_cols].median()
            out[num_cols] = out.groupby(self.time_col)[num_cols].transform(
                lambda s: s.fillna(s.median())
            )
            out[num_cols] = out[num_cols].fillna(global_medians)
            return out

        # Time-safe mode:
        # 1) fill by same-month cross-sectional median
        # 2) remaining NaNs use median from strictly earlier months only
        out[self.time_col] = pd.to_datetime(out[self.time_col], errors="coerce").dt.to_period(
            "M"
        ).dt.to_timestamp()
        original_index = out.index
        sort_cols = [self.time_col]
        if self.id_col in out.columns:
            sort_cols.append(self.id_col)
        out = out.sort_values(sort_cols).copy()

        out[num_cols] = out.groupby(self.time_col)[num_cols].transform(
            lambda s: s.fillna(s.median())
        )

        monthly_medians = (
            out.groupby(self.time_col, sort=True)[num_cols]
            .median()
            .sort_index()
        )
        historical_medians = monthly_medians.expanding(min_periods=1).median().shift(1)

        hist_lookup = historical_medians.add_suffix("__hist")
        out = out.join(hist_lookup, on=self.time_col)
        for col in num_cols:
            hist_col = f"{col}__hist"
            out[col] = out[col].fillna(out[hist_col])
        out = out.drop(columns=list(hist_lookup.columns))

        return out.loc[original_index]

    def drop_low_std_and_high_corr(
        self,
        df: pd.DataFrame,
        char_cols: Optional[List[str]] = None,
        min_std: float = 1e-6,
        max_corr: float = 0.98,
        protected_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Drop characteristics with very low std and very high pairwise correlation.

        Steps:
        1) Remove characteristics with std <= min_std (or NaN std).
        2) On remaining characteristics, greedily remove one variable from each
           pair whose absolute correlation exceeds max_corr.

        Returns
        -------
        filtered_df : pd.DataFrame
        kept_char_cols : list[str]
        dropped_char_cols : list[str]
        """
        if min_std < 0:
            raise ValueError("min_std must be non-negative.")
        if not 0.0 <= max_corr <= 1.0:
            raise ValueError("max_corr must be between 0 and 1.")

        out = df.copy()
        if protected_cols is None:
            protected_cols = [
                self.id_col, self.time_col,
                "ret", "ret_adj", "excess_ret", "year", "y_ipca", "i_idx", "t_idx",
                "prc", "shrout", "cfacpr", "cfacshr", "mcap",
            ]

        if char_cols is None:
            candidate_cols = [
                c for c in out.columns
                if c not in protected_cols and pd.api.types.is_numeric_dtype(out[c])
            ]
        else:
            candidate_cols = [
                c for c in char_cols
                if c in out.columns and pd.api.types.is_numeric_dtype(out[c])
            ]

        if not candidate_cols:
            return out, [], []

        std = out[candidate_cols].std(axis=0, ddof=0)
        low_std_cols = std[(std <= min_std) | (~np.isfinite(std))].index.tolist()
        remaining = [c for c in candidate_cols if c not in low_std_cols]

        dropped_high_corr: List[str] = []
        if remaining:
            corr = out[remaining].corr().abs()
            mean_corr = corr.mean(axis=0).fillna(0.0)
            nan_frac = out[remaining].isna().mean(axis=0)

            work_cols = remaining.copy()
            while len(work_cols) >= 2:
                corr_sub = corr.loc[work_cols, work_cols]
                upper = corr_sub.where(np.triu(np.ones(corr_sub.shape), k=1).astype(bool))
                stacked = upper.stack()
                if stacked.empty:
                    break
                most_corr = stacked.max()
                if not np.isfinite(most_corr) or most_corr <= max_corr:
                    break

                c1, c2 = stacked.idxmax()
                if nan_frac[c1] > nan_frac[c2]:
                    drop_col = c1
                elif nan_frac[c2] > nan_frac[c1]:
                    drop_col = c2
                elif mean_corr[c1] >= mean_corr[c2]:
                    drop_col = c1
                else:
                    drop_col = c2

                dropped_high_corr.append(drop_col)
                work_cols.remove(drop_col)

            remaining = work_cols

        dropped_char_cols = low_std_cols + dropped_high_corr
        keep_set = set(remaining)
        non_char_cols = [c for c in out.columns if c not in candidate_cols]
        final_cols = non_char_cols + [c for c in candidate_cols if c in keep_set]
        return out[final_cols].copy(), remaining, dropped_char_cols

    # ------------------------------------------------------------------
    # Panel construction
    # ------------------------------------------------------------------

    def build_model_panel(
        self,
        df: pd.DataFrame,
        shift_target: bool = True,
        target_col: str = "y_ipca",
        extra_exclude: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Build model-ready panel with target column and integer entity/time ids."""
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

        exclude = {
            self.id_col,
            self.time_col,
            "ret",
            "ret_adj",
            "excess_ret",
            "year",
            target_col,
            "prc",
            "shrout",
            "cfacpr",
            "cfacshr",
            "mcap",
        }
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
        """Normalize using train-only statistics to avoid look-ahead leakage."""
        train_out = train_df.copy()
        test_out = test_df.copy()

        mu = train_out[char_cols].mean(axis=0)
        sd = train_out[char_cols].std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)

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
        """Convert panel DataFrame into X, y, indices arrays ready for InstrumentedPCA."""
        X = panel_df[char_cols].to_numpy(np.float64)
        y = panel_df[target_col].to_numpy(np.float64)
        indices = panel_df[["i_idx", "t_idx"]].to_numpy(np.int64)

        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        return IPCAMatrices(X=X[mask], y=y[mask], indices=indices[mask], char_cols=char_cols)

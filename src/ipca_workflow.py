"""IPCA model fitting, rolling predictions, and Grassmann diagnostics."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from ipca import InstrumentedPCA
from tqdm.auto import tqdm

from src.config import DEFAULT_RFF_GAMMA


@dataclass
class FixedRFFProjection:
    """Fixed RFF projection reused across all rolling windows in one run.

    We keep omega fixed so the feature map is identical for every month.
    Mapping matches the experimentation notebook style:
        z(x) = [sin(x @ omega), cos(x @ omega)]
    """

    omega: np.ndarray  # shape: (n_input_features, n_components)

    @property
    def n_components(self) -> int:
        return int(self.omega.shape[1])

    def transform(self, X: np.ndarray) -> np.ndarray:
        proj = X @ self.omega
        return np.concatenate([np.sin(proj), np.cos(proj)], axis=1)


class IPCAWorkflow:
    """Rolling IPCA estimation and Grassmann-based complexity diagnostics."""

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
    # Core fitting
    # ------------------------------------------------------------------

    @staticmethod
    def fit_ipca(
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        n_factors: int = 3,
        intercept: bool = False,
        max_iter: int = 5000,
        iter_tol: float = 1e-4,
        alpha: float = 1.0,
        silent: bool = True,
        ridge_solver: str = "auto",
        warm_Gamma: Optional[np.ndarray] = None,
        warm_Factors: Optional[np.ndarray] = None,
    ) -> InstrumentedPCA:
        """Fit IPCA model in panel mode.

        Parameters
        ----------
        warm_Gamma : np.ndarray, optional
            Starting values for Gamma (loadings) from a previous fit, e.g.
            last month's ``model.Gamma``.  Speeds up convergence when the
            model is refit on a dataset that only differs by one month.
        warm_Factors : np.ndarray, optional
            Starting values for Factors from a previous fit. Column count must
            match the current training window's number of time periods.
        """
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")

        print(
            f"Fitting IPCA with {n_factors} factors, intercept={intercept}, "
            f"max_iter={max_iter}, iter_tol={iter_tol}, alpha={alpha}, silent={silent}, "
            f"ridge_solver={ridge_solver}, "
            f"warm_Gamma={warm_Gamma is not None}, warm_Factors={warm_Factors is not None}"
        )
        model = InstrumentedPCA(
            n_factors=n_factors,
            intercept=intercept,
            max_iter=max_iter,
            iter_tol=iter_tol,
            alpha=float(alpha),
            l1_ratio=0,
        )
        fit_kwargs: dict = dict(X=X, y=y, indices=indices, ridge_solver=ridge_solver)
        if warm_Gamma is not None:
            fit_kwargs["Gamma"] = warm_Gamma
        if warm_Factors is not None:
            fit_kwargs["Factors"] = warm_Factors

        if silent:
            with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(
                fnull
            ), contextlib.redirect_stderr(fnull):
                model.fit(**fit_kwargs)
        else:
            model.fit(**fit_kwargs)
        return model

    @staticmethod
    def score_ipca(
        model: InstrumentedPCA,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        mean_factor: bool = False,
    ) -> float:
        """Compute model R² score in panel mode."""
        return float(
            model.score(X, y, indices=indices, data_type="panel", mean_factor=mean_factor)
        )

    @staticmethod
    def _predict_ipca_panel(
        model: InstrumentedPCA,
        X: np.ndarray,
        indices: np.ndarray,
        mean_factor: bool = True,
    ) -> np.ndarray:
        """Predict panel returns with compatibility across ipca package signatures."""
        attempts = [
            lambda: model.predict(X, indices=indices, data_type="panel", mean_factor=mean_factor),
            lambda: model.predict(X, indices=indices, data_type="panel"),
            lambda: model.predict(X, indices, data_type="panel", mean_factor=mean_factor),
            lambda: model.predict(X, indices, data_type="panel"),
            lambda: model.predict(X, indices=indices, mean_factor=mean_factor),
            lambda: model.predict(X, indices=indices),
            lambda: model.predict(X, indices),
            lambda: model.predict(X),
        ]
        last_exc: Optional[Exception] = None
        for fn in attempts:
            try:
                pred = fn()
                arr = np.asarray(pred, dtype=np.float64).reshape(-1)
                if arr.shape[0] != X.shape[0]:
                    raise ValueError(
                        f"Unexpected prediction length {arr.shape[0]} for X with {X.shape[0]} rows."
                    )
                return arr
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        raise RuntimeError(
            "Failed to call InstrumentedPCA.predict with known signatures."
        ) from last_exc

    @staticmethod
    def _align_warm_factors(
        prev_factors: np.ndarray,
        prev_dates: np.ndarray,
        curr_dates: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Align prior factors to current dates for factor warm-start."""
        prev_f = np.asarray(prev_factors, dtype=np.float64)
        prev_d = np.asarray(prev_dates).reshape(-1)
        curr_d = np.asarray(curr_dates).reshape(-1)

        if prev_f.ndim != 2 or prev_f.shape[1] != prev_d.size or curr_d.size == 0:
            return None

        out = np.zeros((prev_f.shape[0], curr_d.size), dtype=np.float64)
        matched = np.zeros(curr_d.size, dtype=bool)
        lookup = {d: j for j, d in enumerate(prev_d.tolist())}
        for j, d in enumerate(curr_d.tolist()):
            j_prev = lookup.get(d)
            if j_prev is not None:
                out[:, j] = prev_f[:, j_prev]
                matched[j] = True

        if not matched.any():
            return None
        if not matched.all():
            filler = np.nanmean(out[:, matched], axis=1)
            filler = np.where(np.isfinite(filler), filler, 0.0)
            out[:, ~matched] = filler[:, None]
        return out

    @staticmethod
    def _select_market_cap_universe(
        ids: np.ndarray,
        market_caps: np.ndarray,
        top_n: Optional[int] = None,
        top_share: Optional[float] = None,
        is_log_scale: bool = False,
    ) -> np.ndarray:
        """Select a previous-month stock universe by market-cap rank or share."""
        ids_arr = np.asarray(ids)
        caps_arr = np.asarray(market_caps, dtype=np.float64)
        valid = (~pd.isna(ids_arr)) & np.isfinite(caps_arr)
        ids_arr = ids_arr[valid]
        caps_arr = caps_arr[valid]

        if ids_arr.size == 0:
            return ids_arr[:0]

        order = np.argsort(caps_arr)[::-1]
        ids_arr = ids_arr[order]
        caps_arr = caps_arr[order]

        if top_n is not None:
            keep_n = min(int(top_n), ids_arr.size)
            ids_arr = ids_arr[:keep_n]
            caps_arr = caps_arr[:keep_n]

        if top_share is not None:
            if is_log_scale:
                caps_shifted = caps_arr - np.nanmax(caps_arr)
                weights = np.exp(caps_shifted)
            else:
                weights = np.clip(caps_arr, a_min=0.0, a_max=None)

            total_weight = float(np.nansum(weights))
            if not np.isfinite(total_weight) or total_weight <= 0:
                return ids_arr[:0]

            cutoff_idx = int(
                np.searchsorted(np.cumsum(weights) / total_weight, float(top_share), side="left")
            )
            ids_arr = ids_arr[: cutoff_idx + 1]

        return ids_arr

    @classmethod
    def _build_prev_month_market_cap_universes(
        cls,
        ids_all: np.ndarray,
        month_starts: np.ndarray,
        month_ends: np.ndarray,
        market_caps_all: np.ndarray,
        top_n: Optional[int] = None,
        top_share: Optional[float] = None,
        is_log_scale: bool = False,
    ) -> list[np.ndarray]:
        """Build one stock-universe selection per month using only month t-1 data."""
        universes: list[np.ndarray] = [np.asarray([], dtype=ids_all.dtype) for _ in range(month_starts.size)]
        for month_pos in range(1, month_starts.size):
            prev_slice = slice(month_starts[month_pos - 1], month_ends[month_pos - 1])
            universes[month_pos] = cls._select_market_cap_universe(
                ids=ids_all[prev_slice],
                market_caps=market_caps_all[prev_slice],
                top_n=top_n,
                top_share=top_share,
                is_log_scale=is_log_scale,
            )
        return universes

    # ------------------------------------------------------------------
    # Rolling predictions
    # ------------------------------------------------------------------

    def rolling_ipca_predictions(
        self,
        char_data: pd.DataFrame,
        forecast_start: str | pd.Timestamp = "2016-01-01",
        char_cols: Optional[List[str]] = None,
        target_col: str = "y_ipca",
        n_factors: int = 3,
        intercept: bool = False,
        max_iter: int = 5000,
        iter_tol: float = 1e-4,
        alpha: float = 1.0,
        silent: bool = True,
        ridge_solver: str = "auto",
        normalize: bool = True,
        rolling_normalization: bool = False,
        train_window_months: Optional[int] = None,
        min_train_obs: int = 5000,
        mean_factor: bool = True,
        warm_start: bool = True,
        use_rff: bool = False,
        rff_n_components: int = 500,
        rff_gamma: float = DEFAULT_RFF_GAMMA,
        rff_random_state: Optional[int] = 42,
        rff_omega: Optional[np.ndarray] = None,
        rff_rolling_normalization: Optional[bool] = None,
        market_cap_filter_col: Optional[str] = None,
        market_cap_filter_top_n: Optional[int] = None,
        market_cap_filter_top_share: Optional[float] = None,
        market_cap_filter_is_log: bool = False,
        show_progress: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run rolling IPCA from ``forecast_start`` and return (pred_df, diag_df).

        Refits IPCA each month using only prior data, then predicts for that month.

        Returns
        -------
        pred_df : pd.DataFrame
            Stock-level predictions — one row per (entity, month).
            Columns: id_col, time_col, y_true, y_pred, train_end, n_train, n_test.
        diag_df : pd.DataFrame
            Model-level diagnostics — one row per month.
            Columns: time_col, erank, grassmann_dist, n_train, n_test.

        Parameters
        ----------
        warm_start : bool
            If True (default), the ``Gamma`` (loadings) estimated from the previous
            month's fit are passed as starting values to the current month's fit.
            This soft-starts the ALS iterations close to the previous solution,
            which typically cuts the number of iterations needed for convergence.
        use_rff : bool
            If True, replace original characteristics with fixed RFF features
            before fitting IPCA.
        rff_n_components : int
            Number of random frequency vectors (omega columns). Output feature
            dimension becomes ``2 * rff_n_components`` due to sin/cos concat.
        rff_gamma : float
            Bandwidth/scale multiplier used when sampling omega.
        rff_random_state : int, optional
            RNG seed for deterministic omega sampling.
        rff_omega : np.ndarray, optional
            User-supplied omega matrix of shape
            ``(len(char_cols), rff_n_components)``. If provided, sampling is skipped.
        ridge_solver : str
            Ridge solver passed through to IPCA's Gamma step when ``alpha > 0``.
            Typical values are ``"auto"``, ``"cholesky"``, and ``"lsqr"``.
        rolling_normalization : bool
            If True, recompute train-only normalization statistics at each
            rolling step. If False (default), freeze the normalization
            statistics from the first train window for non-RFF runs. RFF runs
            inherit this setting unless ``rff_rolling_normalization`` is set.
        rff_rolling_normalization : bool, optional
            Optional override for the RFF branch. If True, recompute train-only
            normalization statistics for the RFF branch at each rolling step
            while keeping the sampled ``omega`` fixed. If False, freeze the RFF
            normalization statistics from the first train window so the full
            raw-to-RFF map stays fixed across time. If omitted, the RFF branch
            follows ``rolling_normalization``.
        alpha : float
            Regularization strength for the IPCA Gamma ridge step. Must be >= 0.
        market_cap_filter_col : str, optional
            Column whose previous-month values define the stock universe for
            each rolling fit. The month ``t-1`` cross section is used to select
            stocks for forecast month ``t``, preventing look-ahead leakage.
        market_cap_filter_top_n : int, optional
            Keep the top ``N`` stocks by previous-month market-cap rank.
        market_cap_filter_top_share : float, optional
            Keep the smallest previous-month set whose cumulative market-cap
            share reaches this fraction (for example ``0.5`` for 50%).
        market_cap_filter_is_log : bool
            If True, treat ``market_cap_filter_col`` as a log market-cap proxy
            when computing ``market_cap_filter_top_share``.
        show_progress : bool
            If True, display a month-level progress bar for the rolling refits.
        """
        _empty_pred_cols = [self.id_col, self.time_col, "y_true", "y_pred", "train_end", "n_train", "n_test"]
        if use_rff:
            _empty_pred_cols += ["rff_n_components", "rff_gamma"]
        _empty_diag_cols = [
            self.time_col,
            "erank",
            "grassmann_dist",
            "d_proj",
            "principal_angle_max",
            "principal_angle_mean",
            "geodesic_acceleration",
            "gap_ratio",
            "d_proj_norm",
            "n_train",
            "n_test",
        ]

        work = char_data.copy()
        work[self.time_col] = pd.to_datetime(work[self.time_col], errors="coerce").dt.to_period(
            "M"
        ).dt.to_timestamp()
        work = work.dropna(subset=[self.id_col, self.time_col]).copy()

        if target_col not in work.columns:
            if self.ret_col in work.columns:
                work = work.sort_values([self.id_col, self.time_col]).copy()
                work[target_col] = work.groupby(self.id_col)[self.ret_col].shift(-1)
            else:
                raise ValueError(
                    f"'{target_col}' not found and '{self.ret_col}' unavailable to build it."
                )

        if char_cols is None:
            exclude = {
                self.id_col, self.time_col, self.ret_col,
                "ret", "ret_adj", "excess_ret", "year", target_col, "i_idx", "t_idx",
                "prc", "shrout", "cfacpr", "cfacshr", "mcap",
            }
            char_cols = [
                c for c in work.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(work[c])
            ]
        if not char_cols:
            raise ValueError("No characteristic columns available for IPCA.")
        if market_cap_filter_col is not None and market_cap_filter_col not in work.columns:
            raise ValueError(
                f"market_cap_filter_col '{market_cap_filter_col}' not found in char_data."
            )
        if market_cap_filter_top_n is not None and market_cap_filter_top_share is not None:
            raise ValueError(
                "Specify only one of market_cap_filter_top_n or market_cap_filter_top_share."
            )
        if market_cap_filter_top_n is not None and int(market_cap_filter_top_n) <= 0:
            raise ValueError("market_cap_filter_top_n must be > 0.")
        if market_cap_filter_top_share is not None and not 0.0 < float(market_cap_filter_top_share) <= 1.0:
            raise ValueError("market_cap_filter_top_share must be in (0, 1].")

        work = work.sort_values([self.time_col, self.id_col]).reset_index(drop=True)
        work["i_idx_global"] = work[self.id_col].factorize(sort=True)[0].astype(np.int64)
        work["t_idx_global"] = work[self.time_col].factorize(sort=True)[0].astype(np.int64)

        months = np.sort(work[self.time_col].dropna().unique())
        start_ts = pd.Timestamp(forecast_start).to_period("M").to_timestamp()
        start_np = start_ts.to_datetime64()
        pred_positions = np.flatnonzero(months >= start_np)
        if pred_positions.size == 0:
            return pd.DataFrame(columns=_empty_pred_cols), pd.DataFrame(columns=_empty_diag_cols)

        # Materialize core arrays once and reuse integer slices each month.
        # This avoids repeated DataFrame filtering/copying and preserves strict
        # no-leakage by building train/test sets from month positions only.
        id_all = work[self.id_col].to_numpy()
        time_all = work[self.time_col].to_numpy()
        X_raw_all = work[char_cols].to_numpy(np.float64)
        y_all = work[target_col].to_numpy(np.float64)
        idx_all = work[["i_idx_global", "t_idx_global"]].to_numpy(np.int64)
        market_cap_all = (
            work[market_cap_filter_col].to_numpy(np.float64)
            if market_cap_filter_col is not None else None
        )

        # Rows are contiguous by month because work is sorted by (time, id).
        t_codes = idx_all[:, 1]
        change_points = np.flatnonzero(np.diff(t_codes)) + 1
        month_starts = np.r_[0, change_points]
        month_ends = np.r_[change_points, len(work)]
        if month_starts.size != months.size:
            raise RuntimeError(
                "Month index alignment mismatch. Expected one contiguous block per month."
            )

        prev_month_market_cap_universe = None
        if market_cap_filter_col is not None:
            prev_month_market_cap_universe = self._build_prev_month_market_cap_universes(
                ids_all=id_all,
                month_starts=month_starts,
                month_ends=month_ends,
                market_caps_all=market_cap_all,
                top_n=market_cap_filter_top_n,
                top_share=market_cap_filter_top_share,
                is_log_scale=market_cap_filter_is_log,
            )

        # For each test month, precompute start month position of the rolling train window.
        train_start_pos = np.zeros(months.size, dtype=np.int64)
        if train_window_months is not None:
            months_ns = months.astype("datetime64[ns]")
            for pos, month_val in enumerate(months):
                cutoff = (pd.Timestamp(month_val) - pd.DateOffset(months=train_window_months)).to_datetime64()
                train_start_pos[pos] = int(np.searchsorted(months_ns, cutoff, side="left"))

        effective_rff_rolling_normalization = (
            rolling_normalization
            if rff_rolling_normalization is None
            else bool(rff_rolling_normalization)
        )

        fixed_norm_stats: Optional[tuple[np.ndarray, np.ndarray]] = None

        def _get_fixed_normalization_stats() -> tuple[np.ndarray, np.ndarray]:
            nonlocal fixed_norm_stats
            if fixed_norm_stats is not None:
                return fixed_norm_stats

            first_test_pos = int(pred_positions[0])
            base_start_pos = int(train_start_pos[first_test_pos])
            if base_start_pos >= first_test_pos:
                raise ValueError("Cannot initialize fixed normalization: no base train observations.")

            base_train_slice = slice(month_starts[base_start_pos], month_ends[first_test_pos - 1])
            X_base = X_raw_all[base_train_slice]
            finite_base = np.isfinite(X_base).all(axis=1)
            if prev_month_market_cap_universe is not None:
                base_selected_ids = prev_month_market_cap_universe[first_test_pos]
                finite_base = finite_base & np.isin(id_all[base_train_slice], base_selected_ids)
            if finite_base.sum() == 0:
                raise ValueError("Cannot initialize fixed normalization: no finite base observations.")

            X_base = X_base[finite_base]
            mu = X_base.mean(axis=0)
            sd = X_base.std(axis=0, ddof=0)
            sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
            fixed_norm_stats = (mu, sd)
            return fixed_norm_stats

        # Optional fixed-per-run RFF setup.
        # Omega is always fixed within one call. Normalization can either be
        # fixed from the first train window or recomputed each rolling step.
        rff_proj: Optional[FixedRFFProjection] = None
        rff_mu: Optional[np.ndarray] = None
        rff_sd: Optional[np.ndarray] = None
        if use_rff:
            if rff_gamma <= 0:
                raise ValueError(f"rff_gamma must be > 0, got {rff_gamma}")
            if rff_n_components <= 0:
                raise ValueError(f"rff_n_components must be > 0, got {rff_n_components}")

            if rff_omega is not None:
                omega = np.asarray(rff_omega, dtype=np.float64)
                if omega.ndim != 2:
                    raise ValueError("rff_omega must be a 2D array.")
                if omega.shape[0] != len(char_cols):
                    raise ValueError(
                        f"rff_omega first dimension must equal number of characteristics "
                        f"({len(char_cols)}), got {omega.shape[0]}"
                    )
            else:
                rng = np.random.default_rng(rff_random_state)
                omega = rng.standard_normal((len(char_cols), rff_n_components)) * float(rff_gamma)
            rff_proj = FixedRFFProjection(omega=omega)

            if not effective_rff_rolling_normalization:
                if normalize:
                    rff_mu, rff_sd = _get_fixed_normalization_stats()
                else:
                    rff_mu = np.zeros(len(char_cols), dtype=np.float64)
                    rff_sd = np.ones(len(char_cols), dtype=np.float64)

        # Cache month-level RFF blocks only when normalization is fixed across
        # the entire run; rolling normalization changes the feature map each month.
        rff_cache_X: dict[int, np.ndarray] = {}
        rff_cache_finite: dict[int, np.ndarray] = {}

        def _get_fixed_rff_month_block(month_pos: int) -> tuple[np.ndarray, np.ndarray]:
            X_block = rff_cache_X.get(month_pos)
            finite_block = rff_cache_finite.get(month_pos)
            if X_block is None or finite_block is None:
                month_slice = slice(month_starts[month_pos], month_ends[month_pos])
                X_norm = (X_raw_all[month_slice] - rff_mu) / rff_sd
                X_block = rff_proj.transform(X_norm)
                finite_block = np.isfinite(X_block).all(axis=1)
                rff_cache_X[month_pos] = X_block
                rff_cache_finite[month_pos] = finite_block
            return X_block, finite_block

        results: List[pd.DataFrame] = []
        diag_rows: List[dict] = []
        prev_Gamma: Optional[np.ndarray] = None
        prev_Factors: Optional[np.ndarray] = None
        prev_factor_dates: Optional[np.ndarray] = None
        prev_Gamma_for_dist: Optional[np.ndarray] = None  # Gamma from t-1 for Grassmann distance
        prev_d_proj: Optional[float] = None

        progress_bar = None
        if show_progress:
            progress_bar = tqdm(
                total=int(pred_positions.size),
                desc="Rolling IPCA",
                unit="month",
                leave=True,
            )

        try:
            for offset, test_pos in enumerate(pred_positions, start=1):
                try:
                    test_pos = int(test_pos)
                    test_month = months[test_pos]
                    start_pos = int(train_start_pos[test_pos])
                    if progress_bar is not None:
                        progress_bar.set_postfix_str(
                            f"month={pd.Timestamp(test_month).strftime('%Y-%m')} "
                            f"remaining={pred_positions.size - offset}"
                        )
                    if start_pos >= test_pos:
                        continue

                    train_slice = slice(month_starts[start_pos], month_ends[test_pos - 1])
                    test_slice = slice(month_starts[test_pos], month_ends[test_pos])
                    id_train_raw = id_all[train_slice]
                    y_train_raw = y_all[train_slice]
                    idx_train_raw = idx_all[train_slice]
                    id_test_raw = id_all[test_slice]
                    time_test_raw = time_all[test_slice]
                    idx_test_raw = idx_all[test_slice]
                    y_true_raw = y_all[test_slice]

                    if prev_month_market_cap_universe is not None:
                        selected_ids = prev_month_market_cap_universe[test_pos]
                        if selected_ids.size == 0:
                            continue
                        train_universe_mask = np.isin(id_train_raw, selected_ids)
                        test_universe_mask = np.isin(id_test_raw, selected_ids)
                    else:
                        train_universe_mask = np.ones(y_train_raw.shape[0], dtype=bool)
                        test_universe_mask = np.ones(y_true_raw.shape[0], dtype=bool)

                    if not test_universe_mask.any():
                        continue

                    id_test = id_test_raw[test_universe_mask]
                    time_test = time_test_raw[test_universe_mask]
                    idx_test = idx_test_raw[test_universe_mask]
                    y_true = y_true_raw[test_universe_mask]

                    if use_rff:
                        if effective_rff_rolling_normalization:
                            X_train_raw = X_raw_all[train_slice]
                            finite_train = (
                                train_universe_mask
                                & np.isfinite(X_train_raw).all(axis=1)
                                & np.isfinite(y_train_raw)
                            )
                            if finite_train.sum() < min_train_obs:
                                continue

                            X_train_raw = X_train_raw[finite_train]
                            y_train = y_train_raw[finite_train]
                            idx_train = idx_train_raw[finite_train]

                            if normalize:
                                rff_mu_curr = X_train_raw.mean(axis=0)
                                rff_sd_curr = X_train_raw.std(axis=0, ddof=0)
                                rff_sd_curr = np.where(
                                    np.isfinite(rff_sd_curr) & (rff_sd_curr > 0), rff_sd_curr, 1.0
                                )
                                X_train_norm = (X_train_raw - rff_mu_curr) / rff_sd_curr
                            else:
                                rff_mu_curr = np.zeros(X_train_raw.shape[1], dtype=np.float64)
                                rff_sd_curr = np.ones(X_train_raw.shape[1], dtype=np.float64)
                                X_train_norm = X_train_raw

                            X_train = rff_proj.transform(X_train_norm)

                            X_test_raw = X_raw_all[test_slice][test_universe_mask]
                            X_test_norm = (X_test_raw - rff_mu_curr) / rff_sd_curr
                            X_test = rff_proj.transform(X_test_norm)
                            finite_test = np.isfinite(X_test).all(axis=1)
                        else:
                            # Prune stale months to keep cache bounded under rolling windows.
                            if train_window_months is not None and rff_cache_X:
                                stale = [k for k in rff_cache_X.keys() if k < start_pos]
                                for k in stale:
                                    rff_cache_X.pop(k, None)
                                    rff_cache_finite.pop(k, None)

                            train_feat_blocks: list[np.ndarray] = []
                            train_finite_blocks: list[np.ndarray] = []
                            for month_pos in range(start_pos, test_pos):
                                X_block, finite_block = _get_fixed_rff_month_block(month_pos)
                                train_feat_blocks.append(X_block)
                                train_finite_blocks.append(finite_block)

                            X_train_feat = np.vstack(train_feat_blocks)
                            finite_feat_train = np.concatenate(train_finite_blocks)
                            finite_train = train_universe_mask & finite_feat_train & np.isfinite(y_train_raw)
                            if finite_train.sum() < min_train_obs:
                                continue
                            X_train = X_train_feat[finite_train]
                            y_train = y_train_raw[finite_train]
                            idx_train = idx_train_raw[finite_train]

                            X_test_feat_raw, finite_test_raw = _get_fixed_rff_month_block(test_pos)
                            X_test = X_test_feat_raw[test_universe_mask]
                            finite_test = finite_test_raw[test_universe_mask]
                    else:
                        X_train_raw = X_raw_all[train_slice]
                        finite_train = (
                            train_universe_mask
                            & np.isfinite(X_train_raw).all(axis=1)
                            & np.isfinite(y_train_raw)
                        )
                        if finite_train.sum() < min_train_obs:
                            continue

                        X_train = X_train_raw[finite_train]
                        y_train = y_train_raw[finite_train]
                        idx_train = idx_train_raw[finite_train]

                        X_test_raw = X_raw_all[test_slice][test_universe_mask]
                        finite_test = np.isfinite(X_test_raw).all(axis=1)

                        if normalize:
                            if rolling_normalization:
                                mu = X_train.mean(axis=0)
                                sd = X_train.std(axis=0, ddof=0)
                                sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
                            else:
                                mu, sd = _get_fixed_normalization_stats()
                            X_train = (X_train - mu) / sd
                            X_test = (X_test_raw - mu) / sd
                        else:
                            X_test = X_test_raw

                    # Warm-start: pass previous month's loadings as starting values.
                    # Gamma shape is (L, n_factors) — must match current number of characteristics.
                    # Falls back to cold start on first iteration or if char set changed.
                    use_Gamma = None
                    if warm_start and prev_Gamma is not None:
                        if prev_Gamma.shape[0] == X_train.shape[1]:
                            use_Gamma = prev_Gamma
                    use_Factors = None
                    if warm_start and prev_Factors is not None and prev_factor_dates is not None:
                        curr_train_dates = np.unique(idx_train[:, 1])
                        use_Factors = self._align_warm_factors(
                            prev_factors=prev_Factors,
                            prev_dates=prev_factor_dates,
                            curr_dates=curr_train_dates,
                        )

                    model = self.fit_ipca(
                        X=X_train, y=y_train, indices=idx_train,
                        n_factors=n_factors, intercept=intercept,
                        max_iter=max_iter, iter_tol=iter_tol, alpha=alpha, silent=silent,
                        ridge_solver=ridge_solver,
                        warm_Gamma=use_Gamma,
                        warm_Factors=use_Factors,
                    )

                    # --- Grassmann diagnostics ---
                    curr_Gamma = getattr(model, "Gamma", None)
                    curr_Factors = getattr(model, "Factors", None)
                    model_dates = None
                    if hasattr(model, "metad") and isinstance(model.metad, dict):
                        model_dates = model.metad.get("dates", None)

                    # Effective rank of factor covariance matrix Σ_f = (1/T) F F^T
                    try:
                        _, erank_val, gap_ratio = self.factor_cov_and_erank(model)
                    except Exception:
                        erank_val = np.nan
                        gap_ratio = np.nan

                    # Subspace stability diagnostics relative to the previous fit.
                    gdist = np.nan  # Legacy geodesic distance column
                    d_proj = np.nan
                    angle_max = np.nan
                    angle_mean = np.nan
                    d_proj_norm = np.nan
                    geodesic_acceleration = np.nan
                    if prev_Gamma_for_dist is not None and curr_Gamma is not None:
                        if prev_Gamma_for_dist.shape == curr_Gamma.shape:
                            try:
                                angles = self.principal_angles(curr_Gamma, prev_Gamma_for_dist)
                                if angles.size > 0:
                                    angle_max = float(np.max(angles))
                                    angle_mean = float(np.mean(angles))
                                gdist = self.grassmann_distance(
                                    curr_Gamma, prev_Gamma_for_dist, metric="geodesic"
                                )
                                d_proj = self.grassmann_distance(
                                    curr_Gamma, prev_Gamma_for_dist, metric="projection"
                                )
                                baseline = self.grassmann_random_projection_baseline(
                                    ambient_dim=int(curr_Gamma.shape[0]),
                                    subspace_dim=int(curr_Gamma.shape[1]),
                                )
                                if np.isfinite(baseline) and baseline > 0:
                                    d_proj_norm = float(d_proj / baseline)
                                if prev_d_proj is not None and np.isfinite(prev_d_proj):
                                    geodesic_acceleration = float(d_proj - prev_d_proj)
                            except Exception:
                                gdist = np.nan
                                d_proj = np.nan
                                angle_max = np.nan
                                angle_mean = np.nan
                                d_proj_norm = np.nan
                                geodesic_acceleration = np.nan

                    # Update caches
                    prev_Gamma = curr_Gamma          # warm start for next fit
                    prev_Factors = (
                        np.asarray(curr_Factors, dtype=np.float64)
                        if curr_Factors is not None else None
                    )
                    prev_factor_dates = (
                        np.asarray(model_dates).reshape(-1)
                        if model_dates is not None else None
                    )
                    prev_Gamma_for_dist = curr_Gamma  # distance baseline for next month
                    prev_d_proj = d_proj if np.isfinite(d_proj) else None

                    y_pred = np.full(y_true.shape[0], np.nan, dtype=np.float64)
                    if finite_test.any():
                        y_pred[finite_test] = self._predict_ipca_panel(
                            model=model, X=X_test[finite_test],
                            indices=idx_test[finite_test], mean_factor=mean_factor,
                        )

                    n_rows = int(y_true.shape[0])
                    row = {
                        self.id_col: id_test,
                        self.time_col: time_test,
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "train_end": pd.Timestamp(test_month) - pd.offsets.MonthBegin(1),
                        "n_train": int(X_train.shape[0]),
                        "n_test": int(n_rows),
                    }
                    if use_rff:
                        row["rff_n_components"] = int(rff_proj.n_components)
                        row["rff_gamma"] = float(rff_gamma)
                    results.append(pd.DataFrame(row))

                    # Model-level diagnostics — one row per month
                    diag_rows.append({
                        self.time_col: test_month,
                        "erank": erank_val,
                        "grassmann_dist": gdist,
                        "d_proj": d_proj,
                        "principal_angle_max": angle_max,
                        "principal_angle_mean": angle_mean,
                        "geodesic_acceleration": geodesic_acceleration,
                        "gap_ratio": gap_ratio,
                        "d_proj_norm": d_proj_norm,
                        "n_train": int(X_train.shape[0]),
                        "n_test": int(n_rows),
                    })
                finally:
                    if progress_bar is not None:
                        progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        if not results:
            return pd.DataFrame(columns=_empty_pred_cols), pd.DataFrame(columns=_empty_diag_cols)

        pred_df = pd.concat(results, ignore_index=True).sort_values(
            [self.time_col, self.id_col]
        ).reset_index(drop=True)
        diag_df = pd.DataFrame(diag_rows).sort_values(self.time_col).reset_index(drop=True)
        return pred_df, diag_df

    # ------------------------------------------------------------------
    # OOS R² (from Section 7 of the paper)
    # ------------------------------------------------------------------

    @staticmethod
    def oos_r2(
        pred_df: pd.DataFrame,
        y_true_col: str = "y_true",
        y_pred_col: str = "y_pred",
        benchmark: str = "zero",
    ) -> float:
        """Compute out-of-sample R² from a prediction DataFrame.

        Parameters
        ----------
        benchmark : str
            ``"zero"``  — benchmark is zero (standard in asset pricing, Kelly et al.)
            ``"mean"``  — benchmark is the sample mean of y_true (standard R²)
        """
        df = pred_df.dropna(subset=[y_true_col, y_pred_col])
        y = df[y_true_col].to_numpy(np.float64)
        yhat = df[y_pred_col].to_numpy(np.float64)
        ss_res = np.sum((y - yhat) ** 2)
        if benchmark == "zero":
            ss_tot = np.sum(y ** 2)
        else:
            ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot)

    @staticmethod
    def monthly_oos_r2(
        pred_df: pd.DataFrame,
        time_col: str = "yyyymm",
        y_true_col: str = "y_true",
        y_pred_col: str = "y_pred",
        benchmark: str = "zero",
    ) -> pd.Series:
        """Per-month OOS R², useful for time-series plots."""
        def _r2(g: pd.DataFrame) -> float:
            g = g.dropna(subset=[y_true_col, y_pred_col])
            if g.empty:
                return np.nan
            y = g[y_true_col].to_numpy(np.float64)
            yhat = g[y_pred_col].to_numpy(np.float64)
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum(y ** 2) if benchmark == "zero" else np.sum((y - y.mean()) ** 2)
            return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        return pred_df.groupby(time_col).apply(_r2).rename("oos_r2")

    # ------------------------------------------------------------------
    # Grassmann diagnostics (Section 4 & 7 of the paper)
    # ------------------------------------------------------------------

    @staticmethod
    def effective_rank(matrix: np.ndarray) -> float:
        """Effective (entropic) rank of a positive semi-definite matrix.

        erank(Σ) = exp(H(p))  where p_i = λ_i / tr(Σ) and H is Shannon entropy.
        Equals n when all eigenvalues are equal; approaches 1 when one dominates.
        """
        eigvals = np.linalg.eigvalsh(matrix)
        eigvals = eigvals[eigvals > 0]
        if eigvals.size == 0:
            return 0.0
        p = eigvals / eigvals.sum()
        H = -np.sum(p * np.log(p))
        return float(np.exp(H))

    @staticmethod
    def factor_cov_and_erank(model: InstrumentedPCA) -> tuple[np.ndarray, float, float]:
        """Compute Σ_f, its effective rank, and a retained-factor gap ratio.

        Uses fitted factors stored in ``model.Factors`` and handles both
        common orientations: ``(k, T)`` and ``(T, k)``.
        Returns ``(Sigma_f, erank, gap_ratio)``.

        Since the fitted factor covariance is ``k x k``, the natural
        ``lambda_(k+1)`` noise-floor proxy under a rank-k model is 0, so
        ``gap_ratio = (lambda_k - lambda_(k+1)) / lambda_1`` reduces to
        ``lambda_k / lambda_1``.
        """
        F = np.asarray(model.Factors, dtype=np.float64)
        if F.ndim != 2:
            raise ValueError(f"Expected 2D factor matrix, got shape {F.shape}.")

        k_eff = getattr(model, "n_factors_eff", getattr(model, "n_factors", None))
        # Align to (k, T) before computing Sigma_f.
        if k_eff is not None:
            if F.shape[0] != int(k_eff) and F.shape[1] == int(k_eff):
                F = F.T
        elif F.shape[0] > F.shape[1]:
            # Fallback heuristic when factor count metadata is unavailable.
            F = F.T

        T = max(int(F.shape[1]), 1)
        Sigma_f = (F @ F.T) / T

        eigvals_full = np.linalg.eigvalsh(Sigma_f)
        eigvals_full = np.clip(eigvals_full, 0.0, None)
        if eigvals_full.size == 0 or eigvals_full.max() <= 0:
            return Sigma_f, 0.0, np.nan

        eigvals_desc = np.sort(eigvals_full)[::-1]
        lambda_1 = float(eigvals_desc[0])
        lambda_k = float(eigvals_desc[-1])
        gap_ratio = float(lambda_k / lambda_1) if lambda_1 > 0 else np.nan

        # Numerical guard: drop near-zero eigenvalues before entropy rank.
        tol = np.finfo(np.float64).eps * max(Sigma_f.shape) * lambda_1
        eigvals = eigvals_full[eigvals_full > tol]
        if eigvals.size == 0:
            return Sigma_f, 0.0, gap_ratio
        p = eigvals / eigvals.sum()
        er = float(np.exp(-np.sum(p * np.log(p))))
        return Sigma_f, er, gap_ratio

    @staticmethod
    def _orth_basis(matrix: np.ndarray) -> np.ndarray:
        """Return an orthonormal column basis for the span of ``matrix``."""
        q, _ = np.linalg.qr(np.asarray(matrix, dtype=np.float64), mode="reduced")
        return q

    @classmethod
    def principal_angles(
        cls,
        Gamma1: np.ndarray,
        Gamma2: np.ndarray,
        Z: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Principal angles between two subspaces, in radians."""
        if Z is not None:
            B1 = cls._orth_basis(Z @ Gamma1)
            B2 = cls._orth_basis(Z @ Gamma2)
        else:
            B1 = cls._orth_basis(Gamma1)
            B2 = cls._orth_basis(Gamma2)

        sv = np.linalg.svd(B1.T @ B2, compute_uv=False)
        sv = np.clip(sv, -1.0, 1.0)
        angles = np.arccos(sv)
        return np.sort(angles)[::-1]

    @staticmethod
    def grassmann_random_projection_baseline(
        ambient_dim: int,
        subspace_dim: int,
    ) -> float:
        """Expected projection distance baseline for two random subspaces."""
        m = int(ambient_dim)
        k = int(subspace_dim)
        if m <= 0 or k <= 0 or k >= m:
            return np.nan
        return float(np.sqrt(2.0 * k * (m - k) / (m + 1.0)))

    @classmethod
    def grassmann_distance(
        cls,
        Gamma1: np.ndarray,
        Gamma2: np.ndarray,
        Z: Optional[np.ndarray] = None,
        metric: str = "geodesic",
    ) -> float:
        """Grassmann distance d(V_t, V_{t-1}) between two return subspaces.

        The subspace at each period is V = span(Z @ Gamma^T) ∈ Gr(k, N).
        When Z is not supplied, uses Gamma directly as the basis (i.e. the
        loadings row space, which lives in Gr(k, L)).

        Parameters
        ----------
        Gamma1, Gamma2 : np.ndarray
            Loadings matrices of shape (L, k).
        Z : np.ndarray, optional
            Characteristic matrix of shape (N, L).  If provided, computes
            V = span(Z @ Gamma^T) in R^N (the actual return subspace).
        metric : str
            One of ``"geodesic"``, ``"projection"``, or ``"chordal"``.
        """
        angles = cls.principal_angles(Gamma1, Gamma2, Z=Z)

        if metric == "geodesic":
            return float(np.sqrt(np.sum(angles ** 2)))
        elif metric == "chordal":
            return float(np.sqrt(np.sum(np.sin(angles) ** 2)))
        elif metric == "projection":
            return float(np.sqrt(2.0) * np.sqrt(np.sum(np.sin(angles) ** 2)))
        else:
            raise ValueError(f"Unknown metric '{metric}'. Use 'geodesic', 'chordal', or 'projection'.")

"""IPCA model fitting, rolling predictions, and Grassmann diagnostics."""

from __future__ import annotations

import contextlib
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from ipca import InstrumentedPCA


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
        silent: bool = True,
        warm_Gamma: Optional[np.ndarray] = None,
    ) -> InstrumentedPCA:
        """Fit IPCA model in panel mode.

        Parameters
        ----------
        warm_Gamma : np.ndarray, optional
            Starting values for Gamma (loadings) from a previous fit, e.g.
            last month's ``model.Gamma``.  Speeds up convergence when the
            model is refit on a dataset that only differs by one month.
            Note: Factors are intentionally not warm-started due to a bug in
            the ipca library (passing both Gamma and Factors triggers an
            UnboundLocalError on the first ALS tolerance check).
        """
        print(f"Fitting IPCA with {n_factors} factors, intercept={intercept}, "
              f"max_iter={max_iter}, iter_tol={iter_tol}, silent={silent}, "
              f"warm_start={warm_Gamma is not None}")
        model = InstrumentedPCA(
            n_factors=n_factors,
            intercept=intercept,
            max_iter=max_iter,
            iter_tol=iter_tol,
            alpha=0.01,
            l1_ratio=0.0,
        )
        fit_kwargs: dict = dict(X=X, y=y, indices=indices, data_type="panel")
        if warm_Gamma is not None:
            fit_kwargs["Gamma"] = warm_Gamma
        # Note: do NOT pass Factors — the library has a bug where passing both Gamma
        # and Factors skips SVD initialisation but then references Factor_Old (unset)
        # on the first tolerance check, causing an UnboundLocalError.

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
        silent: bool = True,
        normalize: bool = True,
        train_window_months: Optional[int] = None,
        min_train_obs: int = 5000,
        mean_factor: bool = True,
        warm_start: bool = True,
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
        """
        _empty_pred_cols = [self.id_col, self.time_col, "y_true", "y_pred", "train_end", "n_train", "n_test"]
        _empty_diag_cols = [self.time_col, "erank", "grassmann_dist", "n_train", "n_test"]

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
                "ret", "excess_ret", "year", target_col, "i_idx", "t_idx",
            }
            char_cols = [
                c for c in work.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(work[c])
            ]
        if not char_cols:
            raise ValueError("No characteristic columns available for IPCA.")

        work = work.sort_values([self.time_col, self.id_col]).reset_index(drop=True)
        work["i_idx_global"] = work[self.id_col].factorize(sort=True)[0].astype(np.int64)
        work["t_idx_global"] = work[self.time_col].factorize(sort=True)[0].astype(np.int64)

        months = np.sort(work[self.time_col].dropna().unique())
        start_ts = pd.Timestamp(forecast_start).to_period("M").to_timestamp()
        pred_months = [m for m in months if m >= start_ts]
        if not pred_months:
            return pd.DataFrame(columns=_empty_pred_cols), pd.DataFrame(columns=_empty_diag_cols)

        results: List[pd.DataFrame] = []
        diag_rows: List[dict] = []
        prev_Gamma: Optional[np.ndarray] = None
        prev_Gamma_for_dist: Optional[np.ndarray] = None  # Gamma from t-1 for Grassmann distance

        for test_month in pred_months:
            train = work[work[self.time_col] < test_month].copy()
            if train_window_months is not None:
                cutoff = pd.Timestamp(test_month) - pd.DateOffset(months=train_window_months)
                train = train[train[self.time_col] >= cutoff].copy()

            test = work[work[self.time_col] == test_month].copy()
            if train.empty or test.empty:
                continue

            X_train_raw = train[char_cols].to_numpy(np.float64)
            y_train_raw = train[target_col].to_numpy(np.float64)
            idx_train = train[["i_idx_global", "t_idx_global"]].to_numpy(np.int64)

            finite_train = np.isfinite(X_train_raw).all(axis=1) & np.isfinite(y_train_raw)
            if finite_train.sum() < min_train_obs:
                continue

            X_train = X_train_raw[finite_train]
            y_train = y_train_raw[finite_train]
            idx_train = idx_train[finite_train]

            X_test_raw = test[char_cols].to_numpy(np.float64)
            idx_test = test[["i_idx_global", "t_idx_global"]].to_numpy(np.int64)
            y_true = test[target_col].to_numpy(np.float64)
            finite_test = np.isfinite(X_test_raw).all(axis=1)

            if normalize:
                mu = X_train.mean(axis=0)
                sd = X_train.std(axis=0, ddof=0)
                sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
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

            model = self.fit_ipca(
                X=X_train, y=y_train, indices=idx_train,
                n_factors=n_factors, intercept=intercept,
                max_iter=max_iter, iter_tol=iter_tol, silent=silent,
                warm_Gamma=use_Gamma,
            )

            # --- Grassmann diagnostics ---
            curr_Gamma = getattr(model, "Gamma", None)

            # Effective rank of factor covariance matrix Σ_f = (1/T) F F^T
            try:
                _, erank_val = self.factor_cov_and_erank(model)
            except Exception:
                erank_val = np.nan

            # Grassmann distance d(V_t, V_{t-1}) — NaN for first prediction month
            gdist = np.nan
            if prev_Gamma_for_dist is not None and curr_Gamma is not None:
                if prev_Gamma_for_dist.shape == curr_Gamma.shape:
                    try:
                        gdist = self.grassmann_distance(curr_Gamma, prev_Gamma_for_dist)
                    except Exception:
                        gdist = np.nan

            # Update caches
            prev_Gamma = curr_Gamma          # warm start for next fit
            prev_Gamma_for_dist = curr_Gamma  # distance baseline for next month

            y_pred = np.full(len(test), np.nan, dtype=np.float64)
            if finite_test.any():
                y_pred[finite_test] = self._predict_ipca_panel(
                    model=model, X=X_test[finite_test],
                    indices=idx_test[finite_test], mean_factor=mean_factor,
                )

            n_rows = len(test)
            results.append(pd.DataFrame({
                self.id_col: test[self.id_col].to_numpy(),
                self.time_col: test[self.time_col].to_numpy(),
                "y_true": y_true,
                "y_pred": y_pred,
                "train_end": pd.Timestamp(test_month) - pd.offsets.MonthBegin(1),
                "n_train": int(X_train.shape[0]),
                "n_test": int(n_rows),
            }))

            # Model-level diagnostics — one row per month
            diag_rows.append({
                self.time_col: test_month,
                "erank": erank_val,
                "grassmann_dist": gdist,
                "n_train": int(X_train.shape[0]),
                "n_test": int(n_rows),
            })

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
    def factor_cov_and_erank(model: InstrumentedPCA) -> tuple[np.ndarray, float]:
        """Compute the factor covariance matrix Σ_f and its effective rank.

        Uses fitted factors stored in ``model.Factors`` (shape k × T).
        Returns (Sigma_f, erank).
        """
        F = model.Factors  # shape (k, T)
        T = F.shape[1]
        Sigma_f = (F @ F.T) / T
        er = IPCAWorkflow.effective_rank(Sigma_f)
        return Sigma_f, er

    @staticmethod
    def grassmann_distance(
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
        def _orth_basis(M: np.ndarray) -> np.ndarray:
            """Return orthonormal column basis of span(M)."""
            Q, _ = np.linalg.qr(M, mode="reduced")
            return Q

        if Z is not None:
            B1 = _orth_basis(Z @ Gamma1)   # N × k
            B2 = _orth_basis(Z @ Gamma2)
        else:
            B1 = _orth_basis(Gamma1)        # L × k
            B2 = _orth_basis(Gamma2)

        # Principal angles via SVD of B1^T B2
        sv = np.linalg.svd(B1.T @ B2, compute_uv=False)
        sv = np.clip(sv, -1.0, 1.0)
        angles = np.arccos(sv)

        if metric == "geodesic":
            return float(np.sqrt(np.sum(angles ** 2)))
        elif metric == "chordal":
            return float(np.sqrt(np.sum(np.sin(angles) ** 2)))
        elif metric == "projection":
            return float(np.sqrt(2.0) * np.sqrt(np.sum(np.sin(angles) ** 2)))
        else:
            raise ValueError(f"Unknown metric '{metric}'. Use 'geodesic', 'chordal', or 'projection'.")

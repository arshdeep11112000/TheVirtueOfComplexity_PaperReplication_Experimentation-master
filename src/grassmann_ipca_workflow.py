"""Grassmann-manifold IPCA workflow with the same rolling/no-leakage flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import r2_score

from src.IPCA_Grass_estimator import GrassmannManifoldIPCAEstimator
from src.ipca_workflow import IPCAWorkflow


@dataclass
class GrassmannIPCAResult:
    """Lightweight model wrapper matching the workflow's expected IPCA API."""

    Gamma: np.ndarray
    Factors: np.ndarray  # shape: (k, T)
    history: list[float]
    metad: dict
    estimator: GrassmannManifoldIPCAEstimator
    intercept: bool = False
    n_factors: int = 1
    n_factors_eff: int = 1

    def _design_matrix(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D characteristic matrix, got shape {X_arr.shape}.")

        if self.intercept:
            ones = np.ones((X_arr.shape[0], 1), dtype=np.float64)
            X_arr = np.concatenate([ones, X_arr], axis=1)

        if X_arr.shape[1] != self.Gamma.shape[0]:
            raise ValueError(
                f"Expected {self.Gamma.shape[0]} characteristics after design transform, "
                f"got {X_arr.shape[1]}."
            )
        return X_arr

    def predict(
        self,
        X: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        W: np.ndarray | None = None,
        mean_factor: bool = False,
        data_type: str = "panel",
        label_ind: bool = False,
    ) -> np.ndarray:
        if data_type != "panel":
            raise ValueError("GrassmannIPCAResult only supports panel predictions.")
        if label_ind:
            raise ValueError("label_ind=True is not supported for GrassmannIPCAResult.")
        if W is not None:
            raise ValueError("Portfolio predictions are not supported for GrassmannIPCAResult.")
        if X is None:
            raise ValueError("X must be provided for GrassmannIPCAResult.predict().")

        X_use = self._design_matrix(X)
        factors = np.asarray(self.Factors, dtype=np.float64)
        if factors.ndim != 2:
            raise ValueError(f"Expected 2D factor matrix, got shape {factors.shape}.")

        if mean_factor:
            mean_f = factors.mean(axis=1)
            return np.asarray(X_use @ self.Gamma @ mean_f, dtype=np.float64).reshape(-1)

        if indices is None:
            raise ValueError("indices are required when mean_factor=False.")
        idx = np.asarray(indices)
        if idx.ndim != 2 or idx.shape[0] != X_use.shape[0] or idx.shape[1] < 2:
            raise ValueError(
                f"indices must have shape (n_obs, 2+); got {idx.shape} for {X_use.shape[0]} rows."
            )

        date_codes = idx[:, 1]
        lookup = {int(d): j for j, d in enumerate(np.asarray(self.metad["dates"]).tolist())}
        yhat = np.full(X_use.shape[0], np.nan, dtype=np.float64)
        for d in np.unique(date_codes):
            col = lookup.get(int(d))
            if col is None:
                raise ValueError(
                    "Cannot predict unseen dates with mean_factor=False; "
                    "use mean_factor=True for out-of-sample forecasts."
                )
            mask = date_codes == d
            yhat[mask] = np.asarray(X_use[mask] @ self.Gamma @ factors[:, col], dtype=np.float64)
        return yhat

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        mean_factor: bool = False,
        data_type: str = "panel",
        label_ind: bool = False,
    ) -> float:
        if y is None:
            raise ValueError("y must be provided for GrassmannIPCAResult.score().")
        y_true = np.asarray(y, dtype=np.float64).reshape(-1)
        y_pred = self.predict(
            X=X,
            indices=indices,
            mean_factor=mean_factor,
            data_type=data_type,
            label_ind=label_ind,
        ).reshape(-1)
        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        if finite.sum() == 0:
            return float("nan")
        return float(r2_score(y_true[finite], y_pred[finite]))


class GrassmannIPCAWorkflow(IPCAWorkflow):
    """Rolling IPCA workflow backed by the Grassmann-manifold estimator."""

    def __init__(
        self,
        id_col: str = "permno",
        time_col: str = "yyyymm",
        ret_col: str = "excess_ret",
        optimizer: str = "ConjugateGradient",
        cg_beta_rule: str = "PolakRibiere",
        shrinkage_floor: float = 1e-8,
        optimizer_verbosity: Optional[int] = None,
        log_verbosity: int = 0,
        reuse_line_searcher: bool = True,
    ) -> None:
        super().__init__(id_col=id_col, time_col=time_col, ret_col=ret_col)
        self.optimizer = optimizer
        self.cg_beta_rule = cg_beta_rule
        self.shrinkage_floor = float(shrinkage_floor)
        self.optimizer_verbosity = optimizer_verbosity
        self.log_verbosity = int(log_verbosity)
        self.reuse_line_searcher = bool(reuse_line_searcher)

    @staticmethod
    def _resolve_retraction_method(ridge_solver: str | None) -> str:
        method = "auto" if ridge_solver is None else str(ridge_solver)
        normalized = method.lower().replace("_", "").replace("-", "")
        aliases = {
            "auto": "svd",
            "svd": "svd",
            "polar": "svd",
            "qr": "qr",
            "thinqr": "qr",
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ValueError(
                "For GrassmannIPCAWorkflow, ridge_solver is reused to choose the "
                "manifold retraction and must be one of: auto, svd, polar, qr, thin_qr."
            ) from exc

    @staticmethod
    def _panel_to_grassmann_data(
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
    ) -> tuple[tuple[np.ndarray, np.ndarray], dict]:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        idx_arr = np.asarray(indices, dtype=np.int64)

        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D X, got shape {X_arr.shape}.")
        if y_arr.ndim != 1 or y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError(
                f"Expected y to have shape ({X_arr.shape[0]},), got {y_arr.shape}."
            )
        if idx_arr.ndim != 2 or idx_arr.shape[0] != X_arr.shape[0] or idx_arr.shape[1] < 2:
            raise ValueError(
                f"indices must have shape (n_obs, 2+); got {idx_arr.shape}."
            )

        asset_codes, asset_inv = np.unique(idx_arr[:, 0], return_inverse=True)
        date_codes, date_inv = np.unique(idx_arr[:, 1], return_inverse=True)

        pair_codes = np.ravel_multi_index(
            (date_inv, asset_inv),
            dims=(date_codes.size, asset_codes.size),
        )
        if np.unique(pair_codes).size != X_arr.shape[0]:
            raise ValueError(
                "Duplicate (entity, time) rows detected after filtering; "
                "Grassmann workflow expects one row per asset-month."
            )

        rets = np.zeros((date_codes.size, asset_codes.size), dtype=np.float64)
        Z = np.zeros((date_codes.size, asset_codes.size, X_arr.shape[1]), dtype=np.float64)
        rets[date_inv, asset_inv] = y_arr
        Z[date_inv, asset_inv, :] = X_arr

        metad = {
            "ids": asset_codes,
            "dates": date_codes,
            "N": int(asset_codes.size),
            "T": int(date_codes.size),
            "L": int(X_arr.shape[1]),
        }
        return (rets, Z), metad

    @staticmethod
    def _orthonormalize_columns(W: np.ndarray | None, n_cols: int) -> np.ndarray | None:
        if W is None:
            return None
        W_arr = np.asarray(W, dtype=np.float64)
        if W_arr.ndim != 2 or W_arr.shape[1] != n_cols:
            return None
        q, _ = np.linalg.qr(W_arr, mode="reduced")
        return q[:, :n_cols]

    def fit_ipca(
        self,
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
    ) -> GrassmannIPCAResult:
        del warm_Factors

        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")

        X_fit = np.asarray(X, dtype=np.float64)
        if intercept:
            ones = np.ones((X_fit.shape[0], 1), dtype=np.float64)
            X_fit = np.concatenate([ones, X_fit], axis=1)

        data, metad = self._panel_to_grassmann_data(X_fit, y, indices)
        rets, Z = data

        if Z.shape[2] < n_factors:
            raise ValueError(
                f"Need at least {n_factors} characteristics, got {Z.shape[2]}."
            )

        shrinkage = max(float(alpha), self.shrinkage_floor)
        estimator = GrassmannManifoldIPCAEstimator(
            num_assets=rets.shape[1],
            num_fact=n_factors,
            num_charact=Z.shape[2],
            win_len=rets.shape[0],
            shrinkage=shrinkage,
        )

        initial_point = self._orthonormalize_columns(warm_Gamma, n_factors)
        verbosity = self.optimizer_verbosity
        if verbosity is None:
            verbosity = 0 if silent else 1

        retraction_method = self._resolve_retraction_method(ridge_solver)
        Wopt, f_hat, history = estimator.fit(
            data=data,
            optimizer=self.optimizer,
            max_iterations=max_iter,
            iter_tol=iter_tol,
            verbosity=int(verbosity),
            log_verbosity=self.log_verbosity,
            initial_point=initial_point,
            reuse_line_searcher=self.reuse_line_searcher,
            cg_beta_rule=self.cg_beta_rule,
            retraction_method=retraction_method,
        )

        return GrassmannIPCAResult(
            Gamma=np.asarray(Wopt, dtype=np.float64),
            Factors=np.asarray(f_hat, dtype=np.float64).T,
            history=[float(x) for x in history],
            metad=metad,
            estimator=estimator,
            intercept=intercept,
            n_factors=int(n_factors),
            n_factors_eff=int(n_factors),
        )

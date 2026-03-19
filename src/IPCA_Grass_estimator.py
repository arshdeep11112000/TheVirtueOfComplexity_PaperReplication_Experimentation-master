import numpy as np
from matplotlib import pyplot as plt

import autograd.numpy as anp
from pymanopt import Problem
from pymanopt.manifolds import Grassmann
from pymanopt.function import autograd as pymanopt_autograd
from pymanopt.optimizers import ConjugateGradient, SteepestDescent, TrustRegions


# Summary of what's in this file:
'''
                        ┌─────────────────────────────────────┐
                        │    GRASSMANN IPCA OPTIMIZATION      │
                        └─────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════

  PROBLEM SETUP (done once)
  ─────────────────────────

  Characteristics Z[t]     Loading Matrix W        Factors β_t
     (N × m)          ×      (m × k)         →   solved via OLS
                           ↑ lives on                   ↓
                      Grassmann Gr(m,k)            Λ_t @ β_t = fitted returns
                                                        ↓
                                                 residual = actual - fitted
                                                        ↓
                                              Loss = Σ_t ‖residual_t‖²

═══════════════════════════════════════════════════════════════════

  ITERATIVE OPTIMIZATION (how we search over W's)
  ────────────────────────────────────────────────

  ┌──────────────────────────────────────────────────────────────┐
  │  W₀ = random orthonormal (m × k) matrix on Gr(m,k)         │
  └──────────────────────┬───────────────────────────────────────┘
                         │
           ┌─────────────▼──────────────┐
           │  ITERATION i               │◄──────────────────────┐
           │                            │                       │
           │  1. Evaluate loss f(Wᵢ)    │                       │
           │     (run IPCA for all t)    │                       │
           │                            │                       │
           │  2. Euclidean gradient      │                       │
           │     ∇f(Wᵢ) via autograd    │                       │
           │           │                │                       │
           │           ▼                │                       │
           │  3. Project onto tangent   │                       │
           │     space at Wᵢ:           │                       │
           │     grad = (I-WᵢWᵢᵀ)∇f    │                       │
           │           │                │                       │
           │           ▼                │                       │
           │  4. Conjugate direction:   │                       │
           │     ηᵢ = -grad + β·T(ηᵢ₋₁)│                       │
           │     (parallel transport    │                       │
           │      old direction here)   │                       │
           │           │                │                       │
           │           ▼                │                       │
           │  5. Line search:           │                       │
           │     find step size α       │                       │
           │           │                │                       │
           │           ▼                │                       │
           │  6. Retraction (QR/SVD):   │                       │
           │     Wᵢ₊₁ = retract(Wᵢ+αη) │                       │
           │     → back on manifold!    │                       │
           │           │                │                       │
           └───────────┼────────────────┘                       │
                       │                                        │
                       ▼                                        │
              ┌─────────────────┐     NO                        │
              │  Converged?     ├────────────────────────────────┘
              │  ‖grad‖ < ε ?   │
              └────────┬────────┘
                       │ YES
                       ▼
              ┌─────────────────┐
              │  W* = optimal   │
              │  factor space   │
              └─────────────────┘

══════════════════════════════════════════════════════════════��════

  GEOMETRIC INTUITION
  ───────────────────

     Euclidean space (flat):      Grassmann manifold (curved):

     W ──────────────→            W ·····→
         straight line               ╲  geodesic
         (may leave manifold)          ╲  (stays on surface)
                                        ·→ W_new ∈ Gr(m,k) ✓

'''


#______________________________________________________________________________________________________________


class GrassmannIPCAEstimator:
    def __init__(self, num_assets, num_fact, num_charact, win_len):
        self.num_assets = num_assets  # N
        self.grass_n = num_charact  # m
        self.grass_k = num_fact  # k
        self.dim = self.grass_n * self.grass_k
        self.win_len = win_len

    def loss_fct(self, w, data):
        w = np.asarray(w)
        if w.ndim == 1:
            w = w.reshape(self.grass_n, self.grass_k)
        if w.ndim == 2:
            assert w.shape == (self.grass_n, self.grass_k)

        rets, Z = data
        assert rets.shape == (self.win_len, self.num_assets)
        assert Z.shape == (self.win_len, self.num_assets, self.grass_n)

        obj = 0.0
        for t in range(self.win_len):
            Z_t = Z[t, :, :]  # (N, m)
            Lambda_t = Z_t @ w  # (N, k)
            beta, *_ = np.linalg.lstsq(Lambda_t, rets[t, :], rcond=None)  # (k,)
            fit = Lambda_t @ beta  # (N,)
            resid = rets[t, :] - fit
            obj += resid @ resid

        return obj / self.win_len

    def fit(self, data, max_gen=500):
        w_min = np.full(self.dim, -1.0)
        w_max = np.ones(self.dim)
        pop_size = 5 * self.dim

        de = JDifferentialEvolution(
            w_min, w_max, pop_size,
            model='grassmannian',
            grass_k=self.grass_k
        )

        w_opt, max_dist, max_gen, history = de.optimize(
            self.loss_fct, params=data,
            eps=1e-3, max_gen=max_gen,
            history=True, verbose=True
        )

        W = np.asarray(w_opt).reshape(self.grass_n, self.grass_k)
        print(f"W:\n{W}")
        print(f"max_dist: {max_dist}")
        print(f"max_gen: {max_gen}")
        print(f"objective function: {self.loss_fct(W, data=data)}")

        plt.figure(figsize=(8, 5))
        plt.plot(history, label='Objective function')
        plt.title("IPCA")
        plt.xlabel("Generation (g)")
        plt.ylabel("Objective function (f)")
        plt.legend()
        plt.show()

        return W, history


#______________________________________________________________________________________________________________


class GrassmannManifoldIPCAEstimator:
    """
    IPCA on the Grassmann manifold using pymanopt.

    Optimization variable:
        W = Gamma^T in R^{m x k} with orthonormal columns (Grassmann representative)

    Objective (profiled IPCA loss):
        L(W) = (1/T) sum_t min_f || r_t - (Z_t W) f ||^2

    Post-estimation:
        1) f_hat[t] = argmin_f || r_t - (Z_t W) f ||^2  (cross-sectional LS)
        2) beta_t (optional) = f_hat[t]  (in your codebase, this is the factor return time series)

    Returns (mirrors the GIPCA manifold estimator style):
        W, f_hat, history
    Optionally also returns the raw pymanopt result object.
    """

    def __init__(self, num_assets, num_fact, num_charact, win_len, shrinkage : float = 0.0):
        self.num_assets = num_assets      # N
        self.grass_n = num_charact        # m
        self.grass_k = num_fact           # k
        self.win_len = win_len
        self.shrinkage = shrinkage

    # ------------------------------------------------------------------
    # Post-estimation helper (NumPy)
    # ------------------------------------------------------------------
    def estimate_f(self, W: np.ndarray, data) -> np.ndarray:
        rets, Z = data
        k = self.grass_k
        z = self.shrinkage
        # Vectorised: single batched matmul + solve, no Python loop
        L       = Z @ W                                  # (T, N, k)
        Lt      = L.transpose(0, 2, 1)                   # (T, k, N)
        LtL     = Lt @ L                                  # (T, k, k)
        LtL_reg = LtL + z * np.eye(k)                    # (T, k, k)
        Ltr     = (Lt @ rets[..., None]).squeeze(-1)      # (T, k)
        f_hat   = np.linalg.solve(LtL_reg, Ltr)          # (T, k)
        return f_hat

    # ------------------------------------------------------------------
    # Pymanopt fit (current pymanopt API)
    # ------------------------------------------------------------------
    def fit(
        self,
        data,
        optimizer: str = "ConjugateGradient",
        max_iterations: int = 200,
        verbosity: int = 1,
        log_verbosity: int = 1,
        initial_point: np.ndarray | None = None,
        reuse_line_searcher: bool = False,
        return_pymanopt_result: bool = False,
    ):
        """
        Fit using pymanopt.

        Parameters:
          data: [rets, Z]
              rets: (T, N)
              Z:    (T, N, m)

        Returns:
          Wopt: (m, k)
          f_hat: (T, k)
          history: list of costs (best effort; from result.log if available)
          (optionally) result
        """
        rets, Z = data
        assert rets.shape == (self.win_len, self.num_assets)
        assert Z.shape == (self.win_len, self.num_assets, self.grass_n)
        z_ = float(self.shrinkage) # scalar captured before autograd traces
        k_ = self.grass_k
        # --- objective in autograd.numpy ---
        # NOTE: A Python for-loop is intentionally used here rather than
        # batched matmul/solve.  Benchmarking shows the loop is 1.6-5×
        # FASTER at the actual problem sizes (P∈{1024,4096}, k=24, W=24)
        # because autograd's gradient tape for batched 3-D operations
        # has higher overhead than small 2-D operations in a loop.
        def ipca_profiled_loss_autograd(W, rets_, Z_):
            T_, _ = rets_.shape
            obj = 0.0
            for t in range(T_):
                Zt = Z_[t]               # (N,m)
                rt = rets_[t]            # (N,)
                Lt = anp.dot(Zt, W)      # (N,k)
                XtX = anp.dot(Lt.T, Lt) # (k,k)
                XtX_reg = XtX + z_ * anp.eye(k_) # ridge reg
                Xty = anp.dot(Lt.T, rt) # (k,)
                bt = anp.linalg.solve(XtX_reg, Xty) # ridge β̂
                resid = rt - anp.dot(Lt, bt)
                obj = obj + anp.dot(resid, resid)
            return obj / T_

        manifold = Grassmann(self.grass_n, self.grass_k)
        rets_ag = anp.asarray(rets)
        Z_ag = anp.asarray(Z)

        @pymanopt_autograd(manifold)
        def cost(W):
            return ipca_profiled_loss_autograd(W, rets_ag, Z_ag)

        problem = Problem(manifold=manifold, cost=cost)

        opt = optimizer.lower().replace("_", "")
        if opt in {"cg", "conjugategradient"}:
            solver = ConjugateGradient(
                max_iterations=max_iterations,
                verbosity=verbosity,
                log_verbosity=log_verbosity,
            )
            run_kwargs = {"reuse_line_searcher": reuse_line_searcher}
        elif opt in {"sd", "steepestdescent"}:
            solver = SteepestDescent(
                max_iterations=max_iterations,
                verbosity=verbosity,
                log_verbosity=log_verbosity,
            )
            run_kwargs = {}
        elif opt in {"tr", "trustregions"}:
            solver = TrustRegions(
                max_iterations=max_iterations,
                verbosity=verbosity,
                log_verbosity=log_verbosity,
            )
            run_kwargs = {}
        else:
            raise ValueError("optimizer must be one of: ConjugateGradient, SteepestDescent, TrustRegions")

        if initial_point is not None:
            initial_point = np.asarray(initial_point)
            assert initial_point.shape == (self.grass_n, self.grass_k)
            result = solver.run(problem, initial_point=initial_point, **run_kwargs)
        else:
            result = solver.run(problem, **run_kwargs)

        Wopt = np.asarray(result.point)

        # ---- history extraction (current pymanopt stores it in result.log) ----
        history = []
        if getattr(result, "log", None) is not None:
            iters = result.log.get("iterations", None)
            if isinstance(iters, dict) and "cost" in iters:
                history = [float(c) for c in iters["cost"]]

        if not history:
            history = [float(cost(Wopt))]

        # ---- post-estimation ----
        f_hat = self.estimate_f(Wopt, data)

        if return_pymanopt_result:
            return Wopt, f_hat, history, result
        return Wopt, f_hat, history


#______________________________________________________________________________________________________________


def generate_ipca_data(
        T: int = 252,  # time points
        N: int = 500,  # assets
        m: int = 20,  # TOTAL characteristics (includes intercept if include_intercept=True)
        k: int = 5,  # factors
        seed: int = 124,

        # Factor dynamics
        beta_rho: float = 0.9,  # AR(1) persistence
        sigma_beta: float = 0.5,  # innovation scale

        # Characteristic structure
        z_rho: float = 0.4,  # feature correlation in Toeplitz covariance
        z_scale: float = 1.0,  # overall scale

        # Returns noise
        heavy_tail_df: float = 5.0,  # Student-t dof; set to np.inf for Gaussian
        sigma_eps_base: float = 0.5,  # base idiosyncratic scale
        hetero_strength: float = 0.5,  # cross-sectional heteroskedasticity strength in [0, ~2]

        # Missingness
        missing_prob: float = 0.05,  # per entry missingness probability in Z (non-intercept cols)
        missing_mode: str = "mcAR",  # "mcAR" or "tail" (more missing for extreme values)
        impute: str = "zero",  # "zero" or "mean"

        # Intercept
        include_intercept: bool = True,

        # Optional: time-varying characteristic drift
        z_drift_scale: float = 0.02,
):
    """
    Generates synthetic IPCA-like data with:
      - True W_* on Grassmann (orthonormal columns) in R^{(m(+1)) x k}
      - Cross-sectionally correlated characteristics Z_t (N x m)
      - Optional intercept column (all ones)
      - AR(1) factor returns beta_t
      - Heteroskedastic idiosyncratic noise across assets
      - Optional heavy-tailed noise (Student t)
      - Missingness in characteristics + simple imputation
    Returns:
      data = (rets, Z) where rets is (T,N) and Z is (T,N,m_eff)
      truth dict with W_star, beta, masks, sigmas, etc.
    """

    rng = np.random.default_rng(seed)

    if include_intercept:
        if m < 2:
            raise ValueError("Need m>=2 if include_intercept=True (1 intercept + at least 1 feature).")
        m_eff = m  # total columns returned
        m_core = m - 1  # random feature columns
    else:
        m_eff = m
        m_core = m

    # ----- True W_* (m_eff x k), orthonormal columns -----
    A = rng.normal(size=(m_eff, k))
    W_star, _ = np.linalg.qr(A)  # Grassmann representative

    # ----- Feature covariance for Z core (Toeplitz) -----
    if m_core > 0:
        idx = np.arange(m_core)
        Sigma_z = z_rho ** np.abs(idx[:, None] - idx[None, :])
        Sigma_z = Sigma_z + 1e-10 * np.eye(m_core)
        Lz = np.linalg.cholesky(Sigma_z)
    else:
        Lz = None  # no core features

    # ----- Time-varying drift in characteristic means (core only) -----
    mean_t = np.zeros((T, m_core))
    for t in range(1, T):
        mean_t[t] = mean_t[t - 1] + rng.normal(scale=z_drift_scale, size=m_core)

    # ----- Generate Z (T, N, m_eff) -----
    Z = np.empty((T, N, m_eff), dtype=float)
    for t in range(T):
        if m_core > 0:
            Z_core = (rng.normal(size=(N, m_core)) @ Lz.T) * z_scale
            Z_core = Z_core + mean_t[t]
        else:
            Z_core = np.empty((N, 0), dtype=float)

        if include_intercept:
            Z[t, :, 0] = 1.0
            Z[t, :, 1:] = Z_core
        else:
            Z[t, :, :] = Z_core

    # ----- Factor returns beta_t (T,k): AR(1) -----
    beta = np.zeros((T, k), dtype=float)
    beta[0] = rng.normal(scale=sigma_beta, size=k)
    for t in range(1, T):
        beta[t] = beta_rho * beta[t - 1] + rng.normal(scale=sigma_beta, size=k)

    # ----- Cross-sectional heteroskedastic idiosyncratic scales -----
    u = rng.normal(size=N)
    sigma_i = sigma_eps_base * np.exp(hetero_strength * u)
    sigma_i = np.clip(sigma_i, 1e-4, np.percentile(sigma_i, 99.5))

    # ----- Missingness mask for Z (core columns only; never intercept) -----
    mask = np.ones_like(Z, dtype=bool)

    if missing_prob > 0.0 and m_core > 0:
        core_slice = slice(1, m_eff) if include_intercept else slice(0, m_eff)
        Z_core_all = Z[:, :, core_slice]

        if missing_mode.lower() == "mcar":
            miss = rng.uniform(size=Z_core_all.shape) < missing_prob
        elif missing_mode.lower() == "tail":
            scaled_abs = np.abs(Z_core_all) / (np.std(Z_core_all) + 1e-12)
            p = missing_prob * (1.0 + 0.5 * scaled_abs)
            p = np.clip(p, 0.0, 0.5)
            miss = rng.uniform(size=Z_core_all.shape) < p
        else:
            raise ValueError("missing_mode must be 'mcAR' or 'tail'")

        Z[:, :, core_slice][miss] = np.nan
        mask[:, :, core_slice][miss] = False

        if impute.lower() == "zero":
            Z[:, :, core_slice] = np.nan_to_num(Z[:, :, core_slice], nan=0.0)
        elif impute.lower() == "mean":
            Z_imp = Z[:, :, core_slice]
            for t in range(T):
                col_means = np.nanmean(Z_imp[t], axis=0)
                inds = np.where(~np.isfinite(Z_imp[t]))
                if inds[0].size > 0:
                    Z_imp[t][inds] = col_means[inds[1]]
            Z[:, :, core_slice] = Z_imp
        else:
            raise ValueError("impute must be 'zero' or 'mean'")

    # ----- Generate returns: r_t = (Z_t W_*) beta_t + eps_t -----
    rets = np.empty((T, N), dtype=float)
    use_t = np.isfinite(heavy_tail_df) and heavy_tail_df < 1e9

    for t in range(T):
        Lambda_t = Z[t] @ W_star  # (N,k)
        signal = Lambda_t @ beta[t]  # (N,)

        if use_t:
            df = heavy_tail_df
            if df <= 2:
                raise ValueError("heavy_tail_df must be > 2 for finite variance.")
            eps = rng.standard_t(df, size=N) * np.sqrt((df - 2) / df)
        else:
            eps = rng.normal(size=N)

        rets[t] = signal + sigma_i * eps

    data = [rets, Z]
    truth = {
        "W_star": W_star,
        "beta": beta,
        "sigma_i": sigma_i,
        "mask": mask,
        "m_eff": m_eff,
        "include_intercept": include_intercept,
        "params": {
            "beta_rho": beta_rho,
            "sigma_beta": sigma_beta,
            "z_rho": z_rho,
            "z_scale": z_scale,
            "heavy_tail_df": heavy_tail_df,
            "sigma_eps_base": sigma_eps_base,
            "hetero_strength": hetero_strength,
            "missing_prob": missing_prob,
            "missing_mode": missing_mode,
            "impute": impute,
            "z_drift_scale": z_drift_scale,
        },
    }
    return data, truth


#______________________________________________________________________________________________________________

if __name__ == '__main__':

    # Test code
    seed = 6890
    # seed = 156
    np.random.seed(seed)
    num_assets = 500  # N
    num_fact = 5  # k
    num_charact = 25  # m
    win_len = 63  # T

    # Generate simple data ()
    # rets = np.random.normal(size=(win_len, num_assets))
    # Z = np.random.uniform(size=(win_len, num_assets, num_charact))
    # data = [rets, Z]

    # Generate hard data
    include_intercept = False
    data, truth = generate_ipca_data(T=win_len, N=num_assets, m=num_charact, k=num_fact,
                                      include_intercept=include_intercept, seed=seed)

    # IPCA Manifold estimator
    est_manifold = GrassmannManifoldIPCAEstimator(
        num_assets=num_assets,
        num_fact=num_fact,
        num_charact=num_charact,
        win_len=win_len
    )

    Wopt, f_hat, history = est_manifold.fit(
        data=data,
        optimizer="ConjugateGradient",
        max_iterations=200,
        verbosity=2,
        log_verbosity=1,
    )

    print("Wopt:", Wopt)  # (m, k)
    print("f_hat:", f_hat)  # (T, k)
    print("Final loss:", history[-1])

    # Convergence history
    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.title("Pymanopt convergence (profiled IPCA loss)")
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.show()


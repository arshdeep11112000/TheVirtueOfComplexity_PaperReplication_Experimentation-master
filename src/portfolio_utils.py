"""Portfolio construction helpers for cross-sectional prediction signals."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_quantile_portfolios(
    data: pd.DataFrame,
    score_col: str = "forecast",
    ret_col: str = "excess_ret",
    date_col: str = "yyyymm",
    n_portfolios: int = 10,
    weight_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build monthly quantile portfolios and long-short spread.

    Parameters
    ----------
    data : pd.DataFrame
        Input panel containing at least date, score, and return columns.
    score_col : str
        Column used to sort assets into portfolios.
    ret_col : str
        Realized return column.
    date_col : str
        Date column (monthly); coercible to pandas datetime.
    n_portfolios : int
        Number of portfolios (e.g., 10 for deciles).
    weight_col : str, optional
        If provided, computes value-weighted portfolio returns using this column.
        Otherwise uses equal-weighted returns.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (long_df, wide_df)
        - long_df: one row per (month, portfolio) with portfolio return.
        - wide_df: one row per month with P1...Pn and long_short = Pn - P1.
    """
    required = {date_col, score_col, ret_col}
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if n_portfolios < 2:
        raise ValueError("n_portfolios must be >= 2.")
    if weight_col is not None and weight_col not in data.columns:
        raise ValueError(f"Weight column '{weight_col}' not found.")

    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")
    if weight_col is not None:
        df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")

    keep = [date_col, score_col, ret_col] + ([weight_col] if weight_col else [])
    df = df[keep].dropna(subset=[date_col, score_col, ret_col])
    if df.empty:
        raise ValueError("No valid rows left after coercion/dropna.")

    pct_rank = df.groupby(date_col)[score_col].rank(method="first", pct=True)
    df["portfolio"] = np.ceil(pct_rank * n_portfolios).clip(1, n_portfolios).astype(int)

    if weight_col is None:
        long_df = (
            df.groupby([date_col, "portfolio"], as_index=False)[ret_col]
            .mean()
            .rename(columns={ret_col: "portfolio_return"})
        )
    else:
        tmp = df.dropna(subset=[weight_col]).copy()
        tmp = tmp[tmp[weight_col] > 0]
        if tmp.empty:
            raise ValueError("No positive weights available for weighted portfolio construction.")
        long_df = (
            tmp.groupby([date_col, "portfolio"], as_index=False)
            .apply(
                lambda g: pd.Series(
                    {"portfolio_return": np.average(g[ret_col].to_numpy(), weights=g[weight_col].to_numpy())}
                )
            )
            .reset_index(drop=True)
        )

    wide_df = long_df.pivot(index=date_col, columns="portfolio", values="portfolio_return").sort_index()
    wide_df = wide_df.rename(columns={k: f"P{k}" for k in wide_df.columns})
    wide_df["long_short"] = wide_df[f"P{n_portfolios}"] - wide_df["P1"]
    wide_df = wide_df.reset_index()

    return long_df.sort_values([date_col, "portfolio"]).reset_index(drop=True), wide_df


def build_long_only_portfolio(
    pred_df: pd.DataFrame,
    top_n: int | None = None,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Select stocks each month with positive predicted returns.
    top_n: if set, take only top N stocks by predicted return each month
    threshold: minimum predicted return to be included (default 0)
    """
    required = {"yyyymm", "y_pred", "y_true"}
    missing = [c for c in required if c not in pred_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if top_n is not None and top_n <= 0:
        raise ValueError("top_n must be a positive integer when provided.")

    port = pred_df.copy()
    port["yyyymm"] = pd.to_datetime(port["yyyymm"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    port["y_pred"] = pd.to_numeric(port["y_pred"], errors="coerce")
    port["y_true"] = pd.to_numeric(port["y_true"], errors="coerce")
    port = port.dropna(subset=["yyyymm", "y_pred", "y_true"])
    port = port[port["y_pred"] > threshold].copy()

    if top_n is not None:
        port = (
            port.sort_values(["yyyymm", "y_pred"], ascending=[True, False])
            .groupby("yyyymm", as_index=False, group_keys=False)
            .head(top_n)
            .copy()
        )

    # Equal weight within each month
    month_count = port.groupby("yyyymm")["y_pred"].transform("size")
    port["weight"] = 1.0 / month_count

    return port


def compute_portfolio_returns(port: pd.DataFrame) -> pd.DataFrame:
    """Weighted average realized return each month."""
    required = {"yyyymm", "weight", "y_true"}
    missing = [c for c in required if c not in port.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    monthly = (
        port.groupby("yyyymm", as_index=False)
        .apply(lambda g: pd.Series({"port_ret": (g["weight"] * g["y_true"]).sum()}))
        .reset_index(drop=True)
    )
    return monthly.sort_values("yyyymm").reset_index(drop=True)


def portfolio_performance(monthly_ret: pd.DataFrame, annual_factor: int = 12) -> dict[str, float]:
    required = {"port_ret"}
    missing = [c for c in required if c not in monthly_ret.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    r = pd.to_numeric(monthly_ret["port_ret"], errors="coerce").dropna()
    if r.empty:
        raise ValueError("No valid portfolio returns found.")

    ann_ret = r.mean() * annual_factor
    ann_vol = r.std(ddof=1) * np.sqrt(annual_factor)
    sharpe = np.nan if ann_vol == 0 else ann_ret / ann_vol

    # Max drawdown
    cum = (1 + r).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    max_dd = drawdown.min()

    # Hit rate
    hit_rate = (r > 0).mean()

    print(f"Annualized Return : {ann_ret:.2%}")
    print(f"Annualized Vol    : {ann_vol:.2%}")
    print(f"Sharpe Ratio      : {sharpe:.2f}")
    print(f"Max Drawdown      : {max_dd:.2%}")
    print(f"Hit Rate          : {hit_rate:.2%}")

    return {
        "ann_ret": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe) if not np.isnan(sharpe) else np.nan,
        "max_dd": float(max_dd),
        "hit_rate": float(hit_rate),
    }


def plot_cumulative(monthly_ret: pd.DataFrame) -> None:
    required = {"yyyymm", "port_ret"}
    missing = [c for c in required if c not in monthly_ret.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = monthly_ret.copy()
    data["yyyymm"] = pd.to_datetime(data["yyyymm"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    data["port_ret"] = pd.to_numeric(data["port_ret"], errors="coerce")
    data = data.dropna(subset=["yyyymm", "port_ret"]).sort_values("yyyymm")

    cum = (1 + data["port_ret"]).cumprod()
    plt.figure(figsize=(12, 5))
    plt.plot(data["yyyymm"], cum)
    plt.title("IPCA Long-Only Portfolio — Cumulative Return")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

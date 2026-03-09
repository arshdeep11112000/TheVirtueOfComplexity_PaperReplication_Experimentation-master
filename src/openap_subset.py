"""Helpers for downloading a smaller Open Asset Pricing (OpenAP) panel."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.config import DATA_DIR

META_COLUMNS = {"yyyymm", "permno", "ret", "excess_ret", "year"}


def _rank_factor_relevance(
    data: pd.DataFrame,
    factor_cols: List[str],
    target_col: str,
) -> pd.Series:
    """Rank factors by absolute correlation to target; fallback to coverage."""
    if target_col in data.columns and pd.api.types.is_numeric_dtype(data[target_col]):
        target = data[target_col]
        scores = {}
        for col in factor_cols:
            valid = data[col].notna() & target.notna()
            if valid.sum() < 24:
                scores[col] = 0.0
                continue
            corr = data.loc[valid, col].corr(target.loc[valid])
            scores[col] = float(abs(corr)) if pd.notna(corr) else 0.0
        return pd.Series(scores).sort_values(ascending=False)

    # Fallback: prefer factors with the best data availability.
    return data[factor_cols].notna().mean().sort_values(ascending=False)


def dl_openap_subset(
    n_factors: int = 10,
    n_stocks: int = 20,
    n_years: int = 10,
    target_col: str = "excess_ret",
    backend: str = "pandas",
) -> Tuple[pd.DataFrame, List[str], List[int]]:
    """Download all signals, then keep a compact subset.

    The subset keeps:
    - latest ``n_years`` of data
    - ``n_stocks`` permnos with highest observation count
    - ``n_factors`` factors with highest relevance score
      (absolute correlation with ``target_col`` when available)
    """
    try:
        import openassetpricing as oap
    except ImportError as exc:
        raise ImportError(
            "openassetpricing is required. Install with: pip install openassetpricing"
        ) from exc

    openap = oap.OpenAP()
    data = openap.dl_all_signals(backend)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    if "yyyymm" not in data.columns or "permno" not in data.columns:
        raise ValueError("Expected 'yyyymm' and 'permno' columns in OpenAP data.")

    out = data.copy()
    out["yyyymm"] = pd.to_datetime(out["yyyymm"].astype(str), format="%Y%m", errors="coerce")
    out = out.dropna(subset=["yyyymm", "permno"])
    out["year"] = out["yyyymm"].dt.year

    end_year = int(out["year"].max())
    start_year = end_year - n_years + 1
    out = out[(out["year"] >= start_year) & (out["year"] <= end_year)]

    stock_counts = out.groupby("permno").size().sort_values(ascending=False)
    selected_stocks = stock_counts.head(n_stocks).index.tolist()
    out = out[out["permno"].isin(selected_stocks)]

    factor_cols = [
        c
        for c in out.columns
        if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(out[c])
    ]
    factor_scores = _rank_factor_relevance(out, factor_cols, target_col=target_col)
    selected_factors = factor_scores.head(n_factors).index.tolist()

    keep_cols = [c for c in ["yyyymm", "permno", "ret", "excess_ret"] if c in out.columns]
    keep_cols.extend(selected_factors)

    subset = out[keep_cols].sort_values(["yyyymm", "permno"]).reset_index(drop=True)
    return subset, selected_factors, selected_stocks


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Download a compact OpenAP subset.")
    parser.add_argument("--n-factors", type=int, default=10, help="Number of factors to keep.")
    parser.add_argument("--n-stocks", type=int, default=20, help="Number of stocks to keep.")
    parser.add_argument("--n-years", type=int, default=10, help="Number of latest years to keep.")
    parser.add_argument(
        "--target-col",
        default="excess_ret",
        help="Target column for relevance ranking (default: excess_ret).",
    )
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "openap_subset_10f_20s_10y.parquet"),
        help="Output path (.parquet or .csv).",
    )
    args = parser.parse_args()

    subset, factors, stocks = dl_openap_subset(
        n_factors=args.n_factors,
        n_stocks=args.n_stocks,
        n_years=args.n_years,
        target_col=args.target_col,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        subset.to_csv(output_path, index=False)
    else:
        subset.to_parquet(output_path, index=False)

    print(f"Saved subset to: {output_path}")
    print(f"Shape: {subset.shape}")
    print(f"Selected factors ({len(factors)}): {factors}")
    print(f"Selected stocks ({len(stocks)}): {stocks}")


if __name__ == "__main__":
    main()

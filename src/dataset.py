"""Dataset loading and preprocessing for financial return prediction.

This module provides functions to load and preprocess the Welch-Goyal (2008)
financial predictor dataset and NBER business cycle data.

References:
    - Welch, I., & Goyal, A. (2008). A Comprehensive Look at The Empirical
      Performance of Equity Premium Prediction. Review of Financial Studies.
    - Kelly, B., Malamud, S., & Zhou, K. (2021). The Virtue of Complexity
      in Return Prediction.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import DATA_DIR, NBER_DATA_FILE, PREDICTOR_DATA_FILE

# Feature columns used in the analysis (see footnote [22] in Kelly et al. 2021)
PREDICTOR_COLUMNS = [
    "b/m",      # Book-to-market ratio
    "de",       # Dividend payout ratio (log)
    "dfr",      # Default return spread
    "dfy",      # Default yield spread
    "dp",       # Dividend-price ratio (log)
    "dy",       # Dividend yield (log)
    "ep",       # Earnings-price ratio (log)
    "infl",     # Inflation
    "ltr",      # Long-term return
    "lty",      # Long-term yield
    "ntis",     # Net equity expansion
    "svar",     # Stock variance
    "tbl",      # Treasury bill rate
    "tms",      # Term spread
    "returns",  # Excess returns
]


def load_nber(file_path: Path | None = None) -> pd.DataFrame:
    """Load NBER business cycle dates (peaks and troughs).

    Args:
        file_path: Path to the NBER data file. If None, uses default from config.

    Returns:
        DataFrame with 'peak' and 'trough' columns as datetime objects,
        representing business cycle turning points.

    Raises:
        FileNotFoundError: If the NBER data file is not found.
    """
    if file_path is None:
        file_path = NBER_DATA_FILE

    if not file_path.exists():
        raise FileNotFoundError(f"NBER data file not found: {file_path}")

    nber = pd.read_csv(file_path)[1:]
    nber["peak"] = pd.to_datetime(nber["peak"])
    nber["trough"] = pd.to_datetime(nber["trough"])

    return nber


def load_data(file_path: Path | None = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess the Welch-Goyal financial predictor dataset.

    This function loads monthly financial data and computes derived features
    according to the methodology in Welch and Goyal (2008):
    - dfy: Default yield spread (BAA - AAA corporate bond yields)
    - tms: Term spread (long-term yield - T-bill rate)
    - de: Dividend payout ratio (log dividends - log earnings)
    - dfr: Default return spread (corporate bond return - long-term govt return)
    - dp: Dividend-price ratio (log dividends - log price)
    - dy: Dividend yield (log dividends - log lagged price)
    - ep: Earnings-price ratio (log earnings - log price)

    Args:
        file_path: Path to the predictor data file. If None, uses default from config.

    Returns:
        Tuple containing:
            - features: DataFrame with predictor variables indexed by date
            - returns: Series of monthly returns indexed by date

    Raises:
        FileNotFoundError: If the predictor data file is not found.
    """
    if file_path is None:
        file_path = PREDICTOR_DATA_FILE

    if not file_path.exists():
        raise FileNotFoundError(f"Predictor data file not found: {file_path}")

    # Load raw data
    raw_data = pd.read_csv(file_path)

    # Parse date column
    raw_data["yyyymm"] = pd.to_datetime(
        raw_data["yyyymm"], format="%Y%m", errors="coerce"
    )

    # Clean and convert Index column (remove thousands separator)
    raw_data["Index"] = raw_data["Index"].str.replace(",", "")

    # Set date as index
    raw_data = raw_data.set_index("yyyymm")

    # Convert all columns to float
    raw_data = raw_data.astype(float)

    # Rename Index to prices for clarity
    raw_data = raw_data.rename(columns={"Index": "prices"})

    # Calculate derived features according to Welch and Goyal (2008)
    raw_data["dfy"] = raw_data["BAA"] - raw_data["AAA"]
    raw_data["tms"] = raw_data["lty"] - raw_data["tbl"]
    raw_data["de"] = np.log(raw_data["D12"]) - np.log(raw_data["E12"])
    raw_data["dfr"] = raw_data["corpr"] - raw_data["ltr"]

    lagged_price = raw_data["prices"].shift()
    raw_data["dp"] = np.log(raw_data["D12"]) - np.log(raw_data["prices"])
    raw_data["dy"] = np.log(raw_data["D12"]) - np.log(lagged_price)
    raw_data["ep"] = np.log(raw_data["E12"]) - np.log(raw_data["prices"])

    # Calculate returns (consider using CRSP_SPvw for value-weighted returns)
    raw_data["returns"] = raw_data["prices"].pct_change()
    returns = raw_data["returns"].copy()

    # Select predictor columns and drop rows with missing values
    features = raw_data[PREDICTOR_COLUMNS].dropna()
    returns = returns[returns.index.isin(features.index)]

    return features, returns

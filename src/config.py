"""Configuration settings for the project.

This module centralizes all paths and default parameters used throughout
the codebase, making it easy to modify settings in one place.
"""

from pathlib import Path


# =============================================================================
# Directory Paths
# =============================================================================

# Project root directory (parent of src/)
ROOT_DIR = Path(__file__).parent.parent

# Data directory (source data)
DATA_DIR = ROOT_DIR / "data"

# Cache directory (computed results)
CACHE_DIR = ROOT_DIR / "cache"

# Plots output directory
PLOTS_DIR = ROOT_DIR / "plots"

# Documentation directory
DOCS_DIR = ROOT_DIR / "docs"


# =============================================================================
# Data Files
# =============================================================================

# Welch-Goyal predictor data
PREDICTOR_DATA_FILE = DATA_DIR / "PredictorData2021 - Monthly.csv"

# NBER recession dates
NBER_DATA_FILE = DATA_DIR / "NBER_20210719_cycle_dates_pasted.csv"

# Cached simulation results
METRICS_CACHE_FILE = CACHE_DIR / "metrics.parquet"


# =============================================================================
# Model Default Parameters
# =============================================================================

# Backtest defaults
DEFAULT_TRAIN_WINDOW = 12  # Training window size (T) in months
DEFAULT_RIDGE_LAMBDA = 1000  # Ridge regularization parameter (z)
ANNUALIZATION_FACTOR = 12  # Monthly to annual conversion

# Random Fourier Features defaults
DEFAULT_RFF_GAMMA = 2.0  # Bandwidth parameter
DEFAULT_RFF_N_FEATURES = 6000  # Number of features (P = 2 * n_features)

# Standardization
STANDARDIZATION_BURN_IN = 36  # Months required for initial standardization
VOLATILITY_WINDOW = 12  # Rolling window for return volatility


# =============================================================================
# Simulation Parameters
# =============================================================================

# Default gamma values to test
GAMMA_VALUES = [0.25, 0.5, 1, 2, 4]

# Default ridge lambda values (log-spaced)
Z_VALUES_RANGE = (-3, 6, 10)  # (start, stop, num) for np.logspace

# Number of iterations for RFF variance reduction
DEFAULT_ITERATIONS = 500


# =============================================================================
# Google Drive Configuration
# =============================================================================

# Google Drive folder containing cached metrics
GDRIVE_METRICS_FOLDER_ID = "1SFhFIgPzSsH9FwOBhmkHEoT4016Tld6k"
GDRIVE_METRICS_URL = f"https://drive.google.com/drive/folders/{GDRIVE_METRICS_FOLDER_ID}"


# =============================================================================
# Helper Functions
# =============================================================================

def ensure_dirs_exist() -> None:
    """Create output directories if they don't exist."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_metrics_from_gdrive(force: bool = False) -> Path:
    """Download metrics.parquet from Google Drive if not present locally.

    This function downloads the cached metrics file from the shared Google Drive
    folder. Requires the 'gdown' package to be installed.

    Args:
        force: If True, download even if file already exists locally.

    Returns:
        Path to the downloaded metrics.parquet file.

    Raises:
        ImportError: If gdown is not installed.
        RuntimeError: If download fails.
    """
    ensure_dirs_exist()

    if METRICS_CACHE_FILE.exists() and not force:
        print(f"Metrics file already exists: {METRICS_CACHE_FILE}")
        return METRICS_CACHE_FILE

    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required to download from Google Drive. "
            "Install it with: pip install gdown"
        )

    print(f"Downloading metrics.parquet from Google Drive...")
    print(f"Source: {GDRIVE_METRICS_URL}")

    # Download all files from the folder to cache directory
    gdown.download_folder(
        url=GDRIVE_METRICS_URL,
        output=str(CACHE_DIR),
        quiet=False,
    )

    if not METRICS_CACHE_FILE.exists():
        raise RuntimeError(
            f"Download completed but metrics.parquet not found at {METRICS_CACHE_FILE}. "
            "Please check the Google Drive folder contents."
        )

    print(f"Successfully downloaded to: {METRICS_CACHE_FILE}")
    return METRICS_CACHE_FILE


def load_metrics() -> "pd.DataFrame":
    """Load metrics from cache, downloading from Google Drive if needed.

    Returns:
        DataFrame containing the cached metrics.

    Raises:
        ImportError: If pandas or gdown is not installed.
    """
    import pandas as pd

    if not METRICS_CACHE_FILE.exists():
        download_metrics_from_gdrive()

    return pd.read_parquet(METRICS_CACHE_FILE)

"""The Virtue of Complexity in Return Prediction.

A replication of Kelly, Malamud, & Zhou (2021) demonstrating that
highly parameterized models can outperform simpler ones in return prediction.

Modules:
    config: Configuration settings and paths
    dataset: Data loading and preprocessing for financial data
    rff: Random Fourier Features implementation
    backtest: Ridge regression backtesting framework
"""

from src.backtest import Backtest
from src.config import (
    DATA_DIR,
    DOCS_DIR,
    PLOTS_DIR,
    ROOT_DIR,
    ensure_dirs_exist,
)
from src.dataset import load_data, load_nber
from src.rff import RandomFourierFeatures, RFF

__all__ = [
    "Backtest",
    "DATA_DIR",
    "DOCS_DIR",
    "PLOTS_DIR",
    "ROOT_DIR",
    "RandomFourierFeatures",
    "RFF",
    "ensure_dirs_exist",
    "load_data",
    "load_nber",
]

__version__ = "0.1.0"

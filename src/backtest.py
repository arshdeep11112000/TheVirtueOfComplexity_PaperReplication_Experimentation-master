"""Backtesting framework for ridge regression-based return prediction.

This module implements the backtesting methodology from Kelly, Malamud, & Zhou (2021)
"The Virtue of Complexity in Return Prediction", which demonstrates that highly
parameterized models can outperform simpler ones in return prediction.

The key insight is that the optimal ridge regularization allows models with
complexity ratio c = P/T > 1 (more features than training samples) to achieve
superior out-of-sample performance.
"""

from typing import Any, Dict, Optional, Type

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score, precision_score, r2_score, recall_score

from src.config import (
    ANNUALIZATION_FACTOR,
    DEFAULT_RIDGE_LAMBDA,
    DEFAULT_TRAIN_WINDOW,
)


class Backtest:
    """Rolling-window backtesting for ridge regression return prediction.

    This class implements 1-step ahead prediction using ridge regression with
    a rolling training window, following the methodology in Kelly et al. (2021).

    The model trains on T observations and predicts the next period's return.
    The timing strategy return is computed as: forecast * realized_return.

    Attributes:
        ridge_lambda: Ridge regularization parameter (z in the paper).
        train_window: Number of training samples in the rolling window (T).
        n_features: Number of features in the model (P), set after predict().
        complexity_ratio: Ratio P/T (c in the paper), set after predict().
        backtest_results: DataFrame with prediction results, set after predict().
        prediction: Series of forecasts, set after predict().
        performance_metrics: Dict of performance metrics, set after calc_performance().

    Example:
        >>> bt = Backtest(ridge_lambda=1000, train_window=12)
        >>> bt.predict(features, returns)
        >>> metrics = bt.calc_performance()
        >>> print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    """

    def __init__(
        self,
        ridge_lambda: int = DEFAULT_RIDGE_LAMBDA,
        train_window: int = DEFAULT_TRAIN_WINDOW,
        dtype: Type[np.floating[Any]] = np.float32,
    ) -> None:
        """Initialize the backtesting framework.

        Args:
            ridge_lambda: Ridge regularization parameter. Higher values increase
                regularization strength. In the paper, this is denoted as z.
            train_window: Number of observations in the rolling training window.
                In the paper, this is denoted as T.
            dtype: NumPy dtype for computations. Use float32 for speed or
                float64 for precision.
        """
        self.ridge_lambda = ridge_lambda
        self.train_window = train_window
        self.dtype = dtype

        # Attributes set after predict()
        self.n_features: Optional[int] = None
        self.complexity_ratio: Optional[float] = None
        self.backtest_results: Optional[pd.DataFrame] = None
        self.prediction: Optional[pd.Series] = None

        # Attributes set after calc_performance()
        self.performance_metrics: Optional[Dict[str, float]] = None

    def predict(
        self,
        features: np.ndarray | pd.DataFrame,
        returns: np.ndarray | pd.Series,
    ) -> "Backtest":
        """Run rolling-window ridge regression predictions.

        For each time t from T to T_max, trains on observations [t-T, t) and
        predicts the return at time t. The timing strategy multiplies the
        forecast by the realized return: timing_return = forecast * return.

        Note: Ridge alpha is scaled by T to match the paper's formulation:
            sklearn_alpha = ridge_lambda * T

        Args:
            features: Feature matrix of shape (n_samples, n_features).
                Can be a numpy array or pandas DataFrame.
            returns: Return series of shape (n_samples,).
                Can be a numpy array or pandas Series.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If features and returns have mismatched lengths.
        """
        # Convert to appropriate types if needed
        if isinstance(features, pd.DataFrame):
            feature_values = features.values
            feature_index = features.index
        else:
            feature_values = features
            feature_index = None

        if isinstance(returns, pd.Series):
            return_values = returns.values
            return_index = returns.index
        else:
            return_values = returns
            return_index = feature_index

        n_samples, self.n_features = feature_values.shape
        self.complexity_ratio = self.n_features / self.train_window

        # Validate inputs
        if len(return_values) != n_samples:
            raise ValueError(
                f"features has {n_samples} samples but returns has {len(return_values)}"
            )

        # Initialize ridge model with scaled regularization
        # sklearn's Ridge uses: ||y - Xw||^2 + alpha * ||w||^2
        # Paper uses: ||y - Xw||^2 + z * T * ||w||^2
        ridge_alpha = self.ridge_lambda * self.train_window
        ridge_model = Ridge(alpha=ridge_alpha, solver="svd", fit_intercept=False)

        # Run rolling predictions
        results = []
        prediction_indices = range(self.train_window, n_samples)

        for t in prediction_indices:
            # Training data: [t - T, t)
            train_features = feature_values[t - self.train_window : t].astype(self.dtype)
            train_returns = return_values[t - self.train_window : t].astype(self.dtype)

            # Test data: observation at time t
            test_features = feature_values[t : t + 1].astype(self.dtype)
            test_return = return_values[t : t + 1].astype(self.dtype)

            # Fit model and predict
            ridge_model.fit(train_features, train_returns)
            coefficients = ridge_model.coef_
            forecast = (test_features @ coefficients)[0]

            # Timing strategy: forecast * realized_return
            # This represents a strategy that goes long/short based on forecast magnitude
            timing_return = forecast * test_return[0]

            # Get index for this observation
            obs_index = return_index[t] if return_index is not None else t

            results.append({
                "index": obs_index,
                "coefficient_norm": np.sqrt(np.sum(coefficients ** 2)),
                "forecast": forecast,
                "timing_return": timing_return,
                "market_return": test_return[0],
            })

        # Store results
        self.backtest_results = pd.DataFrame(results).set_index("index")
        self.prediction = self.backtest_results["forecast"]

        return self

    def calc_performance(
        self,
        annualization_factor: int = ANNUALIZATION_FACTOR,
    ) -> Dict[str, float]:
        """Calculate performance metrics for the backtest.

        Computes various metrics including Sharpe ratio, alpha, beta,
        and classification metrics for direction prediction.

        Args:
            annualization_factor: Factor to annualize returns. Use 12 for monthly
                data, 252 for daily data, etc.

        Returns:
            Dictionary containing:
                - coefficient_norm_mean: Average L2 norm of ridge coefficients
                - market_sharpe_ratio: Sharpe ratio of buy-and-hold strategy
                - expected_return: Annualized mean return of timing strategy
                - volatility: Annualized volatility of timing strategy
                - r2: R-squared of forecasts vs realized returns
                - sharpe_ratio: Sharpe ratio of timing strategy
                - information_ratio: Risk-adjusted excess return over market
                - alpha: CAPM alpha of timing strategy
                - precision: Precision of direction prediction
                - recall: Recall of direction prediction
                - accuracy: Accuracy of direction prediction

        Raises:
            RuntimeError: If predict() has not been called first.
        """
        if self.backtest_results is None:
            raise RuntimeError("Must call predict() before calc_performance()")

        data = self.backtest_results.dropna()

        # Fit market model to get alpha and beta
        market_model = LinearRegression().fit(
            data[["market_return"]],
            data["timing_return"],
        )
        strategy_beta = market_model.coef_[0]
        strategy_alpha = market_model.intercept_

        # Annualization factors
        sqrt_factor = np.sqrt(annualization_factor)

        # Timing strategy statistics
        timing_mean = data["timing_return"].mean() * annualization_factor
        timing_std = data["timing_return"].std() * sqrt_factor

        # Market statistics
        market_mean = data["market_return"].mean() * annualization_factor
        market_std = data["market_return"].std() * sqrt_factor

        # Direction prediction (binary classification)
        actual_direction = data["market_return"] > 0
        predicted_direction = data["forecast"] > 0

        self.performance_metrics = {
            "beta_norm_mean": data["coefficient_norm"].mean(),
            "Market Sharpe Ratio": market_mean / market_std,
            "Expected Return": timing_mean,
            "Volatility": timing_std,
            "R2": r2_score(data["market_return"], data["forecast"]),
            "SR": timing_mean / timing_std,
            # Information ratio: excess return over beta-adjusted market, scaled by vol
            "IR": (timing_mean - market_mean * strategy_beta) / timing_std,
            "Alpha": strategy_alpha,
            "Precision": precision_score(actual_direction, predicted_direction),
            "Recall": recall_score(actual_direction, predicted_direction),
            "Accuracy": accuracy_score(actual_direction, predicted_direction),
        }

        return self.performance_metrics

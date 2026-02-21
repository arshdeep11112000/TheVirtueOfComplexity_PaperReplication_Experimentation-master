import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, r2_score, recall_score
import torch
import torch.nn as nn

from src.config import ANNUALIZATION_FACTOR, DEFAULT_TRAIN_WINDOW


class BacktestNN(nn.Module):
    def __init__(
        self,
        d_in: int,
        hidden_dims: int = 64,
        act: str = "relu",
        dropout: float = 0.5,
        T: int = DEFAULT_TRAIN_WINDOW,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
    ):
        super().__init__()
        act = act.lower()
        if act == "relu":
            activation = nn.ReLU()
        elif act == "tanh":
            activation = nn.Tanh()
        elif act == "gelu":
            activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {act}")

        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dims),
            activation,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dims, 1),
        )

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.train_window = T
        self.n_features = d_in
        self.complexity_ratio = d_in / T

        self.backtest_results = None
        self.prediction = None
        self.performance = None
        self.performance_metrics = None
        self.T = T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        epochs: int | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
    ):
        epochs = self.epochs if epochs is None else epochs
        batch_size = self.batch_size if batch_size is None else batch_size
        lr = self.lr if lr is None else lr

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        X_values = X.values if isinstance(X, pd.DataFrame) else X
        y_values = y.values if isinstance(y, pd.Series) else y

        X_tensor = torch.tensor(X_values, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_values, dtype=torch.float32).reshape(-1, 1).to(device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for _ in range(epochs):
            self.train()
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = self(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
    ) -> "BacktestNN" | np.ndarray:
        # Inference mode: keep compatibility with your existing `predict(X)` usage.
        if y is None:
            X_values = X.values if isinstance(X, pd.DataFrame) else X
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)
            X_tensor = torch.tensor(X_values, dtype=torch.float32).to(device)
            self.eval()
            with torch.no_grad():
                predictions = self(X_tensor).cpu().numpy().flatten()
            return predictions

        # Backtest mode: mirror backtest.py's `predict(features, returns)` behavior.
        if isinstance(X, pd.DataFrame):
            feature_values = X.values
            feature_index = X.index
        else:
            feature_values = X
            feature_index = None

        if isinstance(y, pd.Series):
            return_values = y.values
            return_index = y.index
        else:
            return_values = y
            return_index = feature_index

        if feature_values.ndim != 2:
            raise ValueError("features must be 2D with shape (n_samples, n_features)")
        if return_values.ndim != 1:
            raise ValueError("returns must be 1D with shape (n_samples,)")

        n_samples, n_features = feature_values.shape
        self.n_features = n_features
        self.complexity_ratio = self.n_features / self.train_window

        if len(return_values) != n_samples:
            raise ValueError(
                f"features has {n_samples} samples but returns has {len(return_values)}"
            )
        if n_samples <= self.train_window:
            raise ValueError(
                f"Need n_samples > train_window; got n_samples={n_samples}, "
                f"train_window={self.train_window}"
            )

        results = []
        prediction_indices = range(self.train_window, n_samples)

        for t in prediction_indices:
            train_features = feature_values[t - self.train_window : t]
            train_returns = return_values[t - self.train_window : t]
            test_features = feature_values[t : t + 1]
            test_return = return_values[t : t + 1]

            # Refit from scratch each step to match rolling estimation in backtest.py.
            self._reset_parameters()
            self.fit(train_features, train_returns)
            forecast = float(self.predict(test_features)[0])

            coefficient_norm = float(
                np.sqrt(
                    sum(
                        float(torch.sum(param.detach() ** 2).item())
                        for param in self.parameters()
                    )
                )
            )
            timing_return = forecast * float(test_return[0])
            obs_index = return_index[t] if return_index is not None else t

            results.append(
                {
                    "index": obs_index,
                    "coefficient_norm": coefficient_norm,
                    "forecast": forecast,
                    "timing_return": timing_return,
                    "market_return": float(test_return[0]),
                }
            )

        self.backtest_results = pd.DataFrame(results).set_index("index")
        self.prediction = self.backtest_results["forecast"]
        return self

    def calc_performance(self, annualization_factor: int = ANNUALIZATION_FACTOR):
        if self.backtest_results is None:
            raise RuntimeError("Must call predict() before calc_performance()")

        data = self.backtest_results.dropna()
        if data.empty:
            raise RuntimeError("No valid rows in backtest results after dropping NaNs.")

        market_model = LinearRegression().fit(
            data[["market_return"]],
            data["timing_return"],
        )
        strategy_beta = market_model.coef_[0]
        strategy_alpha = market_model.intercept_

        sqrt_factor = np.sqrt(annualization_factor)

        timing_mean = data["timing_return"].mean() * annualization_factor
        timing_std = data["timing_return"].std() * sqrt_factor

        market_mean = data["market_return"].mean() * annualization_factor
        market_std = data["market_return"].std() * sqrt_factor

        actual_direction = data["market_return"] > 0
        predicted_direction = data["forecast"] > 0

        market_sharpe = np.nan if market_std == 0 else market_mean / market_std
        strategy_sharpe = np.nan if timing_std == 0 else timing_mean / timing_std
        information_ratio = (
            np.nan
            if timing_std == 0
            else (timing_mean - market_mean * strategy_beta) / timing_std
        )

        self.performance_metrics = {
            "beta_norm_mean": data["coefficient_norm"].mean(),
            "Market Sharpe Ratio": market_sharpe,
            "Expected Return": timing_mean,
            "Volatility": timing_std,
            "R2": r2_score(data["market_return"], data["forecast"]),
            "SR": strategy_sharpe,
            "IR": information_ratio,
            "Alpha": strategy_alpha,
            "Precision": precision_score(
                actual_direction, predicted_direction, zero_division=0
            ),
            "Recall": recall_score(actual_direction, predicted_direction, zero_division=0),
            "Accuracy": accuracy_score(actual_direction, predicted_direction),
        }

        self.performance = self.performance_metrics["R2"]
        return self.performance_metrics

    def backtest(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.Series | np.ndarray,
    ) -> float:
        self.fit(X, y)
        predictions = self.predict(X_test)
        returns = y_test.values if isinstance(y_test, pd.Series) else y_test
        strategy_returns = predictions * returns
        cumulative_return = np.prod(1 + strategy_returns) - 1
        self.performance = cumulative_return
        return float(cumulative_return)

    def evaluate(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        self.predict(X, y)
        return self.calc_performance()

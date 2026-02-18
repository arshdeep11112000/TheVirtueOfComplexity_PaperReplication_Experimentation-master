
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

class BacktestNN(nn.Module):
    def __init__(self, d_in: int, hidden_dims:int = 64,act : str = "relu", dropout: float = 0.5, T : int = 12):
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
        self.prediction = None
        self.performance = None
        self.T = T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.gamma * (x @ self.W.t()) 
        return torch.cat([torch.sin(z), torch.cos(z)], dim=1)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, epochs: int = 100, batch_size: int = 32, lr: float = 1e-3):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(device)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            self.train()
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = self(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
        self.eval()
        with torch.no_grad():
            predictions = self(X_tensor).cpu().numpy().flatten()
        return predictions
    
    def backtest(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        self.fit(X, y)
        predictions = self.predict(X_test)
        returns = predictions.values
        strategy_returns = predictions * returns
        cumulative_return = np.prod(1 + strategy_returns) - 1
        self.performance = cumulative_return
        return cumulative_return

       
    


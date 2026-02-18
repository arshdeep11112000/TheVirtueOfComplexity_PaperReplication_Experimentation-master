"""Random Fourier Features (RFF) implementation for kernel approximation.

This module implements Random Fourier Features as described in:
    Rahimi, A., & Recht, B. (2007). Random Features for Large-Scale Kernel Machines.

RFF is used to increase dimensionality for the high-complexity regime experiments
in Kelly, Malamud, & Zhou (2021) "The Virtue of Complexity in Return Prediction".
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.config import DEFAULT_RFF_GAMMA, DEFAULT_RFF_N_FEATURES


class RandomFourierFeatures:
    """Transform input features into Random Fourier Features.

    Random Fourier Features provide an explicit mapping that approximates
    the feature map of an RBF kernel, enabling linear methods to capture
    non-linear patterns.

    Attributes:
        gamma: Bandwidth parameter controlling the scale of random frequencies.
        n_features: Number of random Fourier features to generate (output will
            have 2 * n_features dimensions due to sin/cos concatenation).

    Example:
        >>> rff = RandomFourierFeatures(gamma=2.0, n_features=1000)
        >>> features = rff.transform(data, seed=42)
        >>> features.shape
        (n_samples, 2000)
    """

    def __init__(
        self,
        gamma: float = DEFAULT_RFF_GAMMA,
        n_features: int = DEFAULT_RFF_N_FEATURES,
    ) -> None:
        """Initialize the Random Fourier Features transformer.

        Args:
            gamma: Bandwidth parameter for random frequencies. Higher values
                create higher frequency features. Defaults to 2.0.
            n_features: Number of random features to generate. The output
                dimensionality will be 2 * n_features. Defaults to 6000.

        Raises:
            ValueError: If gamma is not a positive number.
            ValueError: If n_features is not a positive integer.
        """
        if not isinstance(gamma, (int, float)) or gamma <= 0:
            raise ValueError(f"gamma must be a positive number, got {gamma}")

        if not isinstance(n_features, int) or n_features <= 0:
            raise ValueError(
                f"n_features must be a positive integer, got {n_features}"
            )

        self.gamma = float(gamma)
        self.n_features = n_features

    def transform(
        self,
        data: pd.DataFrame,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Random Fourier Features for the input data.

        The transformation computes:
            z(x) = [sin(x @ omega), cos(x @ omega)]

        where omega is drawn from N(0, gamma^2 * I).

        Args:
            data: Input data where each row is an observation and each column
                is a feature.
            seed: Random seed for reproducibility. If None, results will vary
                between calls.

        Returns:
            Array of shape (n_samples, 2 * n_features) containing the
            concatenated sine and cosine features.

        Raises:
            ValueError: If data is not a pandas DataFrame.
            ValueError: If data has no columns.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                f"data must be a pandas DataFrame, got {type(data).__name__}"
            )

        n_input_features = len(data.columns)
        if n_input_features == 0:
            raise ValueError("data must have at least one column")

        # Use local random generator for reproducibility without global side effects
        rng = np.random.default_rng(seed)

        # Sample random frequencies from scaled normal distribution
        omega = rng.standard_normal((n_input_features, self.n_features)) * self.gamma

        # Compute projection
        projection = data.values @ omega

        # Compute sine and cosine features
        sine_features = np.sin(projection)
        cosine_features = np.cos(projection)

        # Concatenate and return
        return np.concatenate([sine_features, cosine_features], axis=1)


# Alias for backward compatibility
RFF = RandomFourierFeatures

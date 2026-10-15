"""
Linear and Ridge Regression using NumPy normal equations.

scikit-learn is not used in this model implementation.
"""

from __future__ import annotations

import numpy as np


class NormalEquationRegressor:
    """
    Linear or Ridge Regression using a normal equation.

    alpha=0:
        Ordinary least-squares Linear Regression.

    alpha>0:
        Ridge Regression with L2 regularization.
    """

    def __init__(
        self,
        alpha: float = 0.0,
        fit_intercept: bool = True,
    ):
        if not np.isfinite(float(alpha)) or alpha < 0:
            raise ValueError("alpha cannot be negative")

        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)

        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.n_features_in_: int | None = None

    @staticmethod
    def _validate_X(X) -> np.ndarray:
        """Convert X to a valid floating-point matrix."""
        X_array = np.asarray(X, dtype=float)

        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)

        if X_array.ndim != 2:
            raise ValueError("X must be a two-dimensional array")

        if not np.all(np.isfinite(X_array)):
            raise ValueError("X contains NaN or infinite values")

        return X_array

    def fit(self, X, y) -> NormalEquationRegressor:
        """Estimate the regression coefficients."""
        X_array = self._validate_X(X)
        y_array = np.asarray(y, dtype=float).reshape(-1)

        if len(X_array) == 0 or X_array.shape[1] == 0:
            raise ValueError("Training data cannot be empty")

        if len(X_array) != len(y_array):
            raise ValueError("X and y must have the same row count")

        if not np.all(np.isfinite(y_array)):
            raise ValueError("y contains NaN or infinite values")

        self.n_features_in_ = X_array.shape[1]

        # Centering separates the intercept from coefficient regularization.
        x_mean = X_array.mean(axis=0) if self.fit_intercept else np.zeros(X_array.shape[1])
        y_mean = float(y_array.mean()) if self.fit_intercept else 0.0
        design_matrix = X_array - x_mean
        centered_y = y_array - y_mean
        normal_matrix = design_matrix.T @ design_matrix
        normal_vector = design_matrix.T @ centered_y
        if self.alpha > 0:
            normal_matrix += self.alpha * np.eye(X_array.shape[1])
            self.coef_ = np.linalg.solve(normal_matrix, normal_vector)
            self.solver_ = "ridge_normal_equations"
        elif np.linalg.cond(normal_matrix) < 1e12:
            self.coef_ = np.linalg.solve(normal_matrix, normal_vector)
            self.solver_ = "normal_equations"
        else:
            # Forming X.T@X squares its conditioning. Applying pinv to that
            # matrix can lose useful directions. Use the original design.
            self.coef_ = np.linalg.lstsq(design_matrix, centered_y, rcond=None)[0]
            self.solver_ = "stable_singular_fallback"
        self.intercept_ = float(y_mean - x_mean @ self.coef_)

        return self

    def predict(self, X) -> np.ndarray:
        """Predict continuous target values."""
        if (
            self.coef_ is None
            or self.intercept_ is None
            or self.n_features_in_ is None
        ):
            raise RuntimeError("Call fit before predict")

        X_array = self._validate_X(X)

        if X_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, "
                f"received {X_array.shape[1]}"
            )

        return X_array @ self.coef_ + self.intercept_


class LinearRegressionNormalEquation(NormalEquationRegressor):
    """Ordinary least-squares regression."""

    def __init__(self, fit_intercept: bool = True):
        super().__init__(
            alpha=0.0,
            fit_intercept=fit_intercept,
        )


class RidgeRegressionNormalEquation(NormalEquationRegressor):
    """Ridge Regression with an L2 coefficient penalty."""

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
        )

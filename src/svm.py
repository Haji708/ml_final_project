"""
svm.py — a soft-margin SVM built FROM SCRATCH via PEGASOS (NumPy only).

You do NOT need a QP / SMO solver. Pegasos minimises the regularised
hinge-loss primal by stochastic sub-gradient descent:

    J(w, b) = (lambda/2) * ||w||^2
              + (1/n) * sum_i  max(0, 1 - y_i (w . x_i + b))

with labels y_i in {-1, +1}. Per-step update with learning rate
eta_t = 1 / (lambda * t):

    if  y_i (w . x_i + b) < 1:
        w <- (1 - eta_t*lambda) w + eta_t*y_i*x_i      (hinge active)
    else:
        w <- (1 - eta_t*lambda) w                      (only regularise)

(Handle b with its own sub-gradient step; many implementations fold a bias
feature into x instead — your choice, just be consistent.)

Non-linear extension (pick ONE and justify it in the report):
  - kernelised Pegasos (keep per-sample coefficients alpha_i), OR
  - random Fourier features (map X -> phi(X), then run LINEAR Pegasos).

scikit-learn's SVC is a BASELINE only — not allowed inside this class.
Remember to STANDARDIZE features (see data_prep.standardize) before training.
"""

from __future__ import annotations

import numpy as np


class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,     # regularisation strength (1/(n*C) flavour)
        n_iters: int = 20_000,     # number of sub-gradient steps
        random_state: int = 42,
        feature_map: str = "linear",  # feature map for non-linear SVM (default: linear)
        n_components: int = 500,  # number of random Fourier features (if applicable)
        gamma: float = 1.0,  # RBF kernel parameter (if applicable)
        record_every: int = 100,  # record objective every N iterations
    ):
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.random_state = random_state
        self.w = None              # weight vector
        self.b = 0.0               # bias
        self.objective_history_ = []
        self.objective_iterations_ = []
        self.margin_width_ = None
        self.support_indices_ = None
        self.support_vectors_ = None
        self.support_margins_ = None
        self.feature_map = feature_map
        self.n_components = n_components
        self.gamma = gamma
        self.record_every = record_every

        if lambda_ <= 0:
            raise ValueError("lambda_ must be positive")
        if n_iters <= 0:
            raise ValueError("n_iters must be positive")
        if feature_map not in {"linear", "rff"}:
            raise ValueError("feature_map must be 'linear' or 'rff'")
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if record_every <= 0:
            raise ValueError("record_every must be positive")

    @staticmethod
    def _to_pm1(y):
        """Map {0,1} (or any 2-class labels) to {-1, +1}. Store the mapping."""
        y = np.array(y)
        classes = np.unique(y)

        if len(classes) != 2:
            raise ValueError("y must have exactly 2 unique classes")
        y_pm1 = np.where(y == classes[0], -1, 1)
        return y_pm1, classes

    def _validate_inputs(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if X.shape[0] == 0:
            raise ValueError("X and y cannot be empty")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")

        return X, y

    def _transform_features(self, X):
        X = np.asarray(X, dtype=np.float64)

        if self.feature_map == "linear":
            return X

        projection = X @ self.rff_weights_ + self.rff_offset_

        return np.sqrt(2.0 / self.n_components) * np.cos(projection)

    def _objective(self, X, y_pm1):
        scores = X @ self.w + self.b
        hinge_loss = np.maximum(0.0, 1.0 - y_pm1 * scores)

        return (
                0.5 * self.lambda_ * np.dot(self.w, self.w)
                + np.mean(hinge_loss)
        )

    def fit(self, X, y):
        """
        Run Pegasos. Sample one (or a mini-batch of) example(s) per step,
        use eta_t = 1/(lambda*t), and apply the update above.
        Return self.
        """
        X, y = self._validate_inputs(X, y)
        y_pm1, self.classes_ = self._to_pm1(y)

        self.n_features_in_ = X.shape[1]

        self._fit_feature_map(X.shape[1])
        X_transformed = self._transform_features(X)

        self.X_train_ = X.copy()
        # RFF matrices can be large; the local transform is already owned by
        # this fit, so an additional full-size copy is unnecessary.
        self.X_train_transformed_ = X_transformed
        self.y_pm1_ = y_pm1.copy()

        self.w = np.zeros(X_transformed.shape[1])
        self.b = 0.0

        # Initialize the random number generator
        rng = np.random.default_rng(self.random_state)

        self.objective_history_ = []
        self.objective_iterations_ = []
        for t in range(1, self.n_iters + 1):
            # Sample a random index
            i = rng.integers(X_transformed.shape[0])
            x_i = X_transformed[i]
            y_i = y_pm1[i]

            # Compute the learning rate
            eta_t = 1 / (self.lambda_ * t)

            # Compute the decision value
            decision_value = np.dot(self.w, x_i) + self.b
            margin = y_i * decision_value

            # Update weights and bias based on hinge loss condition
            if margin < 1:
                self.w = (1 - eta_t * self.lambda_) * self.w + eta_t * y_i * x_i
                self.b += eta_t * y_i
            else:
                self.w = (1 - eta_t * self.lambda_) * self.w

            if (
                    t == 1
                    or t % self.record_every == 0
                    or t == self.n_iters
            ):
                self.objective_iterations_.append(t)
                self.objective_history_.append(
                    self._objective(X_transformed, y_pm1)
                )

        scores = X_transformed @ self.w + self.b
        margins = y_pm1 * scores

        weight_norm = np.linalg.norm(self.w)

        self.margin_width_ = (
            np.inf if weight_norm == 0.0 else 2.0 / weight_norm
        )

        support_mask = margins <= 1.0 + 1e-6
        self.support_indices_ = np.flatnonzero(support_mask)
        self.support_vectors_ = X[self.support_indices_]
        self.support_margins_ = margins[self.support_indices_]

        return self

    def decision_function(self, X):
        """Return the signed margin  w . x + b  for each row."""
        if self.w is None:
            raise ValueError("Model has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")

        if not hasattr(self, "n_features_in_"):
            raise ValueError("Model has not been fitted correctly")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected input with {self.n_features_in_} features, "
                f"got {X.shape[1]}."
            )

        X_transformed = self._transform_features(X)
        return X_transformed @ self.w + self.b

    def predict(self, X):
        """Return class labels (map sign of decision_function back to your labels)."""
        scores = self.decision_function(X)
        pm1_predictions = np.where(scores >= 0, 1, -1)

        return np.where(
            pm1_predictions == 1,
            self.classes_[1],
            self.classes_[0]
        )

    def _fit_feature_map(self, n_features):
        if self.feature_map == "linear":
            return

        rng = np.random.default_rng(self.random_state)

        self.rff_weights_ = np.sqrt(2.0 * self.gamma) * rng.normal(
            size=(n_features, self.n_components)
        )
        self.rff_offset_ = rng.uniform(
            0.0,
            2.0 * np.pi,
            size=self.n_components,
        )


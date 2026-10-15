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
    ):
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.random_state = random_state
        self.w = None              # weight vector
        self.b = 0.0               # bias

    @staticmethod
    def _to_pm1(y):
        """Map {0,1} (or any 2-class labels) to {-1, +1}. Store the mapping."""
        y = np.array(y)
        classes = np.unique(y)

        if len(classes) != 2:
            raise ValueError("y must have exactly 2 unique classes")
        y_pm1 = np.where(y == classes[0], -1, 1)
        return y_pm1, classes

    def fit(self, X, y):
        """
        Run Pegasos. Sample one (or a mini-batch of) example(s) per step,
        use eta_t = 1/(lambda*t), and apply the update above.
        Return self.
        """
        X = np.asanyarray(X, dtype=np.float64)
        y = np.asanyarray(y)
        y_pm1, self.classes = self._to_pm1(y)

        self.w = np.zeros(X.shape[1])
        self.b = 0.0

        rng = np.random.default_rng(self.random_state)

        for t in range(1, self.n_iters + 1):
            # Sample a random index
            i = rng.integers(X.shape[0])
            x_i = X[i]
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
        return self

    def decision_function(self, X):
        """Return the signed margin  w . x + b  for each row."""
        if self.w is None:
            raise ValueError("Model has not been fitted yet.")
        X = np.asanyarray(X, dtype=np.float64)
        if X.shape[1] != self.w.shape[0]:
            raise ValueError(f"Expected input with {self.w.shape[0]} features, got {X.shape[1]}.")
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        """Return class labels (map sign of decision_function back to your labels)."""
        scores = self.decision_function(X)
        pm1_predictions = np.where(scores >= 0, 1, -1)

        return np.where(
            pm1_predictions == 1,
            self.classes[1],
            self.classes[0]
        )

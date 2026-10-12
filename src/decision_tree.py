"""
decision_tree.py — a decision tree built FROM SCRATCH (NumPy only).

Must support BOTH tasks:
  - classification (criterion = "gini" or "entropy")  -> price tier
  - regression     (criterion = "mse")                -> price

scikit-learn is NOT allowed here. This is your own code.

Suggested design:
  - a small Node (feature index, threshold, left, right, value/leaf_value),
  - recursive _grow with a best-split search over features & thresholds,
  - impurity functions (gini / entropy / mse) and the impurity-decrease
    objective used to pick splits,
  - hyperparameters that control overfitting (below).
"""
from __future__ import annotations
import pandas as pd
import numpy as np


class Node:
    """One node of the tree. Fill in the fields you need."""
    def __init__(self):
        self.feature = None      # int: feature index to split on (internal)
        self.threshold = None    # float: split threshold (internal)
        self.left = None         # Node
        self.right = None        # Node
        self.value = None        # leaf prediction (class label/proba or mean)

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(
        self,
        task: str = "classification",   # "classification" | "regression"
        criterion: str = "gini",        # "gini" | "entropy" | "mse"
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
    ):
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root: Node | None = None

    # --- impurity -----------------------------------------------------------

    def _impurity(self, y) -> float:
        """Gini / entropy (classification) or MSE/variance (regression)."""
        #if self.criterion not in ['mse','gini','entropy']:
        if self.criterion == 'mse':
            return np.var(y)
        if self.criterion == 'gini':
            if len(y)==0:
                return 0.0
            classes, counts  = np.unique(y, return_counts=True)
            probs = counts/len(y)
            return 1-np.sum(probs**2)
        if self.criterion == 'entropy':
            if len(y)==0:
                return 0.0
            classes, counts  = np.unique(y, return_counts=True)
            probs = counts/len(y)
            return -np.sum(probs*np.log2(probs))


    def optimized_bilateral_impurity(self, y):
        """Gini / entropy (classification) or MSE/variance (regression)."""
        #if self.criterion not in ['mse','gini','entropy']:
        if len(y)==0:
                return 0.0

        if self.criterion == 'mse':
            left_sum_y = np.cumsum(y)
            prefix_sum_y_sqr =np.cumsum(y**2)

            y_sum = np.sum(y)
            y_sum_sq = np.sum(y** 2)

            left_sum_y = left_sum_y[:-1]
            right_sum_y = y_sum - left_sum_y

            left_sums_sq = np.cumsum(y** 2)[:-1]
            right_sums_sq = y_sum_sq - left_sums_sq

            left_counts = np.arange(1, len(y))
            right_counts = len(y) - left_counts

            left_vars = (left_sums_sq / left_counts) - (left_sums / left_counts) ** 2
            right_vars = (right_sums_sq / right_counts) - (right_sums / right_counts) ** 2
            return left_vars, right_vars
        if self.criterion == 'gini':
            
            classes, counts  = np.unique(y, return_counts=True)
            probs = counts/len(y)
            return 1-np.sum(probs**2)
        if self.criterion == 'entropy':
            classes, counts  = np.unique(y, return_counts=True)
            probs = counts/len(y)
            return -np.sum(probs*np.log2(probs))

            


        raise NotImplementedError("TODO: impurity")

    def get_optimal_threshold(self,x, y):
        
        sorted_indices = np.argsort(x)
        y_sorted = y[sorted_indices]
        x_sorted = x[sorted_indices]
        parent_impurity = self._impurity(y_sorted)
        n = len(y_sorted)
        left_impurity,right_impurity = self.optimized_bilateral_impurity(y_sorted)
        left_counts = np.arange(1, n)
        right_counts = n - left_counts
        weighted_child_impurities = (left_counts / n) * left_impurity + (right_counts / n) * right_impurity
        gains = parent_impurity - weighted_child_impurities
        best_idx = np.argmax(gains)
        return (x_sorted[best_idx] + x_sorted[best_idx + 1]) / 2.0

    def _best_split(self, X, y):
        """
        Search features & candidate thresholds; return the split that
        maximizes impurity decrease (feature, threshold, decrease), or
        None if no valid split improves things.
        """
        
                



        raise NotImplementedError("TODO: best split search")

    # --- grow / predict -----------------------------------------------------
    def _grow(self, X, y, depth: int) -> Node:
        """Recursively build the tree, respecting the stopping rules."""
        raise NotImplementedError("TODO: recursive grow")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.root = self._grow(X, y, depth=0)
        return self

    def _predict_one(self, x, node: Node):
        raise NotImplementedError("TODO: traverse to a leaf")

    def predict(self, X):
        raise NotImplementedError("TODO: predict")

    def predict_proba(self, X):
        """Classification only: return class probabilities from leaf counts."""
        raise NotImplementedError("TODO: predict_proba (classification)")

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
    def __init__(self,feature=None,threshold=None,left=None,right=None,value=None):
        self.feature = feature      # int: feature index to split on (internal)
        self.threshold = threshold    # float: split threshold (internal)
        self.left = left         # Node
        self.right = right        # Node
        self.value = value        # leaf prediction (class label/proba or mean)

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


    def optimized_bilateral_impurity(self, y_sorted):
        """Gini / entropy (classification) or MSE/variance (regression)."""
        #if self.criterion not in ['mse','gini','entropy']:
        if len(y_sorted)==0:
                return 0.0

        if self.criterion == 'mse':
            left_sum_y = np.cumsum(y_sorted)
            prefix_sum_y_sqr =np.cumsum(y_sorted**2)

            y_sum = np.sum(y_sorted)
            y_sum_sq = np.sum(y_sorted** 2)

            left_sum_y = left_sum_y[:-1]
            right_sum_y = y_sum - left_sum_y

            left_sums_sq = np.cumsum(y_sorted** 2)[:-1]
            right_sums_sq = y_sum_sq - left_sums_sq

            left_counts = np.arange(1, len(y_sorted))
            right_counts = len(y_sorted) - left_counts

            left_vars = (left_sums_sq / left_counts) - (left_sums / left_counts) ** 2
            right_vars = (right_sums_sq / right_counts) - (right_sums / right_counts) ** 2
            return left_vars, right_vars
        if self.criterion == 'gini':
            
            classes = np.unique(y_sorted)
            n_classes = len(classes)
            n = len(y_sorted)
            y_onehot = (y_sorted[:, None] == classes).astype(int)

            total_class_counts = np.sum(y_onehot, axis=0)

            left_class_counts = np.cumsum(y_onehot, axis=0)[:-1]
            right_class_counts = total_class_counts - left_class_counts

            left_counts = np.arange(1, n)[:, None]
            right_counts = n - left_counts

            gini_left = 1 - np.sum((left_class_counts/left_counts)**2, axis = 1)
            gini_right = 1 - np.sum((right_class_counts/right_counts)**2, axis = 1)

            return gini_left,gini_right
        if self.criterion == 'entropy':
            classes = np.unique(y_sorted)
            n_classes = len(classes)
            n = len(y_sorted)
            y_onehot = (y_sorted[:, None] == classes).astype(int)

            total_class_counts = np.sum(y_onehot, axis=0)

            left_counts = np.arange(1, n)[:, None]
            right_counts = n - left_counts

            left_class_counts = np.cumsum(y_onehot, axis=0)[:-1]
            right_class_counts = total_class_counts - left_class_counts

    

            entropy_left =  - np.sum((left_class_counts/left_counts) * np.log2(left_class_counts/left_counts+ 1e-9) , axis = 1) # adding a small number to avoid log2(0)
            entropy_right =  - np.sum((right_class_counts/right_counts) * np.log2(right_class_counts/right_counts+ 1e-9) , axis = 1)
            return entropy_left,entropy_right
            


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

        valid_splits = (left_counts >= self.min_samples_leaf) & (right_counts >= self.min_samples_leaf)

        if not np.any(valid_splits):
            return None, None

        weighted_child_impurities = (left_counts / n) * left_impurity + (right_counts / n) * right_impurity
        gains = parent_impurity - weighted_child_impurities
        best_idx = np.argmax(gains)
        if gains[best_idx] <= 0:
            return None, -1.0
        return [(x_sorted[best_idx] + x_sorted[best_idx + 1]) / 2.0 , gains[best_idx]] # categorical features to be handled later

    def _best_split(self, X, y):
        """
        Search features & candidate thresholds; return the split that
        maximizes impurity decrease (feature, threshold, decrease), or
        None if no valid split improves things.
        """
        n_features = X.shape[1]
        
        thresholds = np.zeros(n_features)
        gains = np.full(n_features, -1.0) 

        for col_idx in range(n_features):
            thresh, gain = self.get_optimal_threshold(X[:, col_idx], y)
            if thresh is not None:
                thresholds[col_idx] = thresh
                gains[col_idx] = gain

        
        if np.max(gains) <= 0:
            return None, None

        best_col = np.argmax(gains)
        best_threshold = thresholds[best_col]

        return best_col, best_threshold



        raise NotImplementedError("TODO: best split search")

    def compute_leaf_value(self,y):
        if self.task=='regression':
            return np.mean(y)
        else:
            
            counts = np.array([np.sum(y == c) for c in self.classes_])
            probs = counts / len(y)
            
            majority_label = self.classes_[counts.argmax()]
            
            return probs, majority_label

    # --- grow / predict -----------------------------------------------------
    def _grow(self, X, y,depth: int,valid_indices=None) -> Node:
        """Recursively build the tree, respecting the stopping rules."""
        if valid_indices is None:
            valid_indices = np.arange(len(y))
        sample_size = len(valid_indices)

        if depth>=self.max_depth or sample_size<self.min_samples_split:
            return Node(None,None,None,None,self.compute_leaf_value(y[valid_indices])) 
        best_col, best_threshold = self._best_split(X[valid_indices], y[valid_indices])

        if best_col is None:
            return Node(None,None,None,None,self.compute_leaf_value(y[valid_indices]))

        feat_vals = X[valid_indices, best_col]
        left_indices = valid_indices[np.where(feat_vals<=best_threshold)[0]]
        right_indices = valid_indices[np.where(feat_vals>best_threshold)[0]]

        left_child = self._grow(X,y,depth+1,left_indices)
        right_child = self._grow(X,y,depth+1,right_indices)

        return Node(best_col,best_threshold,left_child,right_child,None)

        raise NotImplementedError("TODO: recursive grow")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.root = self._grow(X, y, depth=0)
        return self

    def _predict_one(self, x, node: Node):
        print(f"Visiting node -> Feature: {node.feature}, Threshold: {node.threshold}, Value: {node.value}, Is Leaf: {node.is_leaf()}")
        if node.is_leaf():
            return node.value[1] if self.task=='classification' else node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x,node.left)
        return self._predict_one(x,node.right)

        raise NotImplementedError("TODO: traverse to a leaf")

    def predict(self, X):
        return np.array([self._predict_one(row,self.root) for row in X])

        raise NotImplementedError("TODO: predict")

    def predict_proba_one(self, x,node:Node):
        """Classification only: return class probabilities from leaf counts."""
        if node.is_leaf():
            return node.value[0]
        if x[node.feature] <= node.threshold:
            return self.predict_proba_one(x,node.left)
        return self.predict_proba_one(x,node.right)

        
        
    def predict_proba(self, X):
        """Classification only: return class probabilities from leaf counts."""
        return np.array([self.predict_proba_one(row,self.root) for row in X])
        raise NotImplementedError("TODO: predict_proba (classification)")



"""
X = np.array([
    [2.0, 3.0],  
    [1.0, 1.5], 
    [8.0, 8.5],  
    [9.0, 9.5]   
])

y = np.array([0, 0, 1, 1])

tree = DecisionTree(task='classification',criterion='gini',max_depth=10)
tree.fit(X,y)
print(tree.predict(X))
print('_______________')
print(tree.predict_proba(X))
"""
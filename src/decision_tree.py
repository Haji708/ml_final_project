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
import data_prep
import time

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
        max_depth: int | None = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
    ):  
        if task  not in ["classification" , "regression"]:
            raise ValueError("Invalid task")
        if task == 'regression' and criterion!='mse':
            raise ValueError("Criterion for regression task can only be mse")
        if task == 'classification' and criterion not in ['gini','entropy']:
            raise ValueError("Criterion for classification task can only be either gini or entropy")
        
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

            left_vars = (left_sums_sq / left_counts) - (left_sum_y / left_counts) ** 2
            right_vars = (right_sums_sq / right_counts) - (right_sum_y / right_counts) ** 2
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

    def get_optimal_threshold(self, x, y):
        if(len(x)!=len(y)):
            raise ValueError("Shape mismatch in feature column 'x' and value column 'y")
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]
        n = len(y_sorted)

        if n < 2:
            return None, -1.0

        left_impurity, right_impurity = self.optimized_bilateral_impurity(y_sorted)
        left_counts = np.arange(1, n)
        right_counts = n - left_counts

        valid = (
            (x_sorted[:-1] < x_sorted[1:])
            & (left_counts >= self.min_samples_leaf)
            & (right_counts >= self.min_samples_leaf)
        )
        candidate_indices = np.flatnonzero(valid)

        if candidate_indices.size == 0:
            return None, -1.0

        gains = self._impurity(y_sorted) - (
            (left_counts[candidate_indices] / n) * left_impurity[candidate_indices]
            + (right_counts[candidate_indices] / n) * right_impurity[candidate_indices]
        )

        best_position = int(np.argmax(gains))
        best_index = candidate_indices[best_position]

        if gains[best_position] <= 0:
            return None, -1.0

        threshold = (x_sorted[best_index] + x_sorted[best_index + 1]) / 2.0
        return float(threshold), float(gains[best_position])

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
        if len(left_indices) == 0 or len(right_indices) == 0:
            return Node(None, None, None, None, self.compute_leaf_value(y[valid_indices]))

        left_child = self._grow(X,y,depth+1,left_indices)
        right_child = self._grow(X,y,depth+1,right_indices)

        return Node(best_col,best_threshold,left_child,right_child,None)

        raise NotImplementedError("TODO: recursive grow")

    def fit(self, X, y):
        if X.shape[0]!=len(y):
            raise ValueError("Shape mismatch in data matrix 'X' and value column 'y")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.root = self._grow(X, y, depth=0)
        return self

    def _predict_one(self, x, node: Node):
        #print(f"Visiting node -> Feature: {node.feature}, Threshold: {node.threshold}, Value: {node.value}, Is Leaf: {node.is_leaf()}")
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
        if self.task == 'regression':
            raise ValueError("predict_proba is not supported for regression tasks.")
        """Classification only: return class probabilities from leaf counts."""
        if node.is_leaf():
            return node.value[0]
        if x[node.feature] <= node.threshold:
            return self.predict_proba_one(x,node.left)
        return self.predict_proba_one(x,node.right)

        
        
    def predict_proba(self, X):
        if self.task == 'regression':
            raise ValueError("predict_proba is not supported for regression tasks.")
        """Classification only: return class probabilities from leaf counts."""
        return np.array([self.predict_proba_one(row,self.root) for row in X])
        raise NotImplementedError("TODO: predict_proba (classification)")




"""
prepared = data_prep.prepare_data()
X = prepared["X_train"]
y = prepared["y_train"]
X_test =prepared["X_test"]
y_test =prepared["y_test"]
start_time = time.perf_counter()
tree = DecisionTree(task='regression',criterion='mse',max_depth=10)
tree.fit(X,y)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")
y_pred=tree.predict(X_test)

print('_______________')
mae = (sum(abs(y_pred[i]-y_test[i]) for i in range(len(y_pred)))/len(y_test))
baseline_mae = np.mean(np.abs(y_test - np.median(y)))
print("Model MAE:", mae)
print("Median-baseline MAE:", baseline_mae)
print("Test median price:", np.median(y_test)
"""
# depth =15 ,  23s ; depth =5 ,7s ; depth = 10 ,13s
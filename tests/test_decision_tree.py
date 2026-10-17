import pytest
import numpy as np
import sys
from pathlib import Path


import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


from decision_tree import DecisionTree,Node 


@pytest.fixture
def binary_classification_data():
    """Fixture providing a tiny synthetic dataset for testing."""
    X_train = np.array([
        [2.0, 3.0],  
        [1.0, 1.5],  
        [8.0, 8.5],  
        [9.0, 9.5]   
    ])
    y_train = np.array([0, 0, 1, 1])
    return X_train, y_train


def test_classification_predictions(binary_classification_data):
    """Test that the tree correctly fits and predicts labels on training data."""
    X_train, y_train = binary_classification_data
    tree = DecisionTree(max_depth=3, min_samples_split=2, task='classification')
    tree.fit(X_train, y_train)
    
    preds = tree.predict(X_train)
    np.testing.assert_array_equal(preds, y_train)


def test_classification_probabilities(binary_classification_data):
    """Test that predict_proba returns a properly shaped 2D matrix summing to 1.0."""
    X_train, y_train = binary_classification_data
    tree = DecisionTree(max_depth=3, min_samples_split=2, task='classification')
    tree.fit(X_train, y_train)
    
    probs = tree.predict_proba(X_train)
    
    # Check shape: (n_samples, n_classes) -> (4, 2)
    assert probs.shape == (4, 2)
    
    # Check that class probabilities for each row sum up to 1.0
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0)


def test_unseen_data_predictions(binary_classification_data):
    """Test generalization on new, unseen data points."""
    X_train, y_train = binary_classification_data
    tree = DecisionTree(max_depth=3, min_samples_split=2, task='classification')
    tree.fit(X_train, y_train)
    
    X_test = np.array([
        [1.2, 1.8],  # Should predict Class 0
        [8.5, 9.0]   # Should predict Class 1
    ])
    
    preds = tree.predict(X_test)
    np.testing.assert_array_equal(preds, [0, 1])


def test_shape_mismatch_error(binary_classification_data):
    """Test that mismatched X and y lengths raise an exception during fit."""
    X_train, _ = binary_classification_data
    tree = DecisionTree(max_depth=3, task='classification')
    
    # X has 4 rows, but we pass only 2 labels
    with pytest.raises(Exception):
        tree.fit(X_train, np.array([0, 1]))


def test_task_criterion_mismatch(binary_classification_data):
    """Test that calling predict_proba on a regression tree raises a ValueError."""
    X_train, _ = binary_classification_data
    

    with pytest.raises(ValueError, match="Criterion for regression task can only be mse"):
        reg_tree = DecisionTree(max_depth=3, task='regression')
    


def test_non_numeric_input_error(binary_classification_data):
    """Test that passing non-numeric (string) features raises a conversion error."""
    _, y_train = binary_classification_data
    bad_X = np.array([["apple", "banana"], ["orange", "grape"], ["pear", "peach"], ["kiwi", "melon"]])
    bad_tree = DecisionTree(max_depth=3, task='classification')
    
    with pytest.raises(ValueError):
        bad_tree.fit(bad_X, y_train)
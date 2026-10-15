"""Tests for Member 1's regression and metric implementations."""

import unittest

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

from src import evaluate
from src.linear_regression import (
    LinearRegressionNormalEquation,
    RidgeRegressionNormalEquation,
)


class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        self.X = np.array(
            [
                [1.0, 2.0],
                [2.0, 0.0],
                [3.0, 1.0],
                [4.0, 3.0],
                [5.0, 2.0],
            ]
        )
        self.y = (
            5.0
            + 2.0 * self.X[:, 0]
            - 3.0 * self.X[:, 1]
        )

    def test_known_coefficients(self):
        model = LinearRegressionNormalEquation()
        model.fit(self.X, self.y)

        np.testing.assert_allclose(
            model.coef_,
            np.array([2.0, -3.0]),
            atol=1e-8,
        )
        self.assertAlmostEqual(model.intercept_, 5.0)

    def test_matches_sklearn(self):
        custom = LinearRegressionNormalEquation()
        baseline = LinearRegression()

        custom.fit(self.X, self.y)
        baseline.fit(self.X, self.y)

        np.testing.assert_allclose(
            custom.predict(self.X),
            baseline.predict(self.X),
            atol=1e-8,
        )

    def test_ridge_matches_sklearn(self):
        alpha = 2.0

        custom = RidgeRegressionNormalEquation(alpha=alpha)
        baseline = Ridge(alpha=alpha)

        custom.fit(self.X, self.y)
        baseline.fit(self.X, self.y)

        np.testing.assert_allclose(
            custom.predict(self.X),
            baseline.predict(self.X),
            atol=1e-8,
        )


class TestRegressionMetrics(unittest.TestCase):
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])

        self.assertEqual(evaluate.rmse(y, y), 0.0)
        self.assertEqual(evaluate.mae(y, y), 0.0)
        self.assertEqual(evaluate.r2(y, y), 1.0)

    def test_known_errors(self):
        true = np.array([1.0, 2.0, 3.0])
        predicted = np.array([2.0, 2.0, 2.0])

        self.assertAlmostEqual(
            evaluate.rmse(true, predicted),
            np.sqrt(2.0 / 3.0),
        )
        self.assertAlmostEqual(
            evaluate.mae(true, predicted),
            2.0 / 3.0,
        )
        self.assertAlmostEqual(
            evaluate.r2(true, predicted),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
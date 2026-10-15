"""Additional behavior tests for the issues found during the real-data review."""
import unittest
import warnings
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from src import data_prep, evaluate
from src.linear_regression import LinearRegressionNormalEquation, RidgeRegressionNormalEquation
from src.member1_regression import linear_feature_matrix, _diagnostic_cv


def sample_frame():
    return data_prep.clean(pd.DataFrame({"price": [100000, 200000, 300000, 400000],
        "estate_id": ["a", "b", "c", "d"], "Sahə": ["80 m²", "100 m²", "2 sot", "100 sot"],
        "Kateqoriya": ["Yeni tikili", "Yeni tikili", "Torpaq", "Torpaq"],
        "city": ["a", "b", "a", "b"], "location": ["north", "south", "north", "south"],
        "Mərtəbə": ["2 / 10", "3 / 10", None, None], "Otaq sayı": [2, 3, None, None],
        "views": [10, 20, 30, 40], "Torpaq sahəsi": [None, None, "2 sot", "100 sot"]}))


class DataReviewTests(unittest.TestCase):
    def test_land_sot_converted_in_both_columns(self):
        frame = sample_frame()
        processor = data_prep.TabularPreprocessor().fit(frame)
        X, _ = processor.transform(frame)
        self.assertEqual(len(frame), 4)  # 2 sot is not wrongly excluded as <5 m².
        j = processor.feature_names_.index("Sahə_m2")
        np.testing.assert_allclose(X[:, j], [80, 100, 200, 10000])
        k = processor.feature_names_.index("Torpaq_sahəsi_m2")
        np.testing.assert_allclose(X[:, k], [0, 0, 200, 10000])

    def test_missing_land_has_indicator(self):
        frame = sample_frame(); p = data_prep.TabularPreprocessor().fit(frame)
        X, _ = p.transform(frame)
        np.testing.assert_array_equal(X[:, p.feature_names_.index("Torpaq_missing")], [1, 1, 0, 0])

    def test_latest_id_is_by_timestamp_not_row_order(self):
        raw = pd.DataFrame({"price": [200000, 100000], "estate_id": ["x", "x"],
                           "Sahə": ["80 m²"] * 2, "Kateqoriya": ["Yeni tikili"] * 2,
                           "datetime_scrape_y": ["2024-11-10", "2024-10-10"]})
        self.assertEqual(data_prep.clean(raw).price.tolist(), [200000])

    def test_non_azn_is_not_silently_used_as_azn(self):
        raw = pd.DataFrame({"price": [100000], "Sahə": ["80 m²"],
                            "Kateqoriya": ["Yeni tikili"], "currency_x": ["USD"]})
        with self.assertRaises(ValueError): data_prep.clean(raw)

    def test_leakage_columns_are_excluded_but_not_required(self):
        raw = pd.DataFrame({"price": [100000], "Sahə": ["80 m²"], "Kateqoriya": ["Yeni tikili"],
                           "unit_price": [1250], "total_price": [100000], "owner_name": ["private"]})
        result = data_prep.clean(raw)
        self.assertFalse({"unit_price", "total_price", "owner_name"} & set(result.columns))
        data_prep.clean(raw.drop(columns=["unit_price", "total_price"]))

    def test_training_imputation_unchanged_by_held_out_rows(self):
        frame = sample_frame(); p = data_prep.TabularPreprocessor().fit(frame.iloc[:2])
        before = dict(p.numeric_medians_)
        held_out = frame.iloc[2:].copy(); held_out["Sahə"] = ["1000000 m²", None]
        X, _ = p.transform(held_out)
        self.assertEqual(before, p.numeric_medians_)
        self.assertEqual(X[1, p.feature_names_.index("Sahə_m2")], 90)

    def test_unknown_location_preserves_column_order(self):
        frame = sample_frame(); p = data_prep.TabularPreprocessor().fit(frame.iloc[:2])
        held_out = frame.iloc[2:].copy(); held_out["location"] = "new location"
        X, _ = p.transform(held_out)
        columns = [i for i, c in enumerate(p.feature_names_) if c.startswith("location=")]
        self.assertEqual(X.shape[1], len(p.feature_names_))
        self.assertTrue((X[:, columns] == 0).all())

    def test_feature_helper_requires_fitted_state(self):
        with self.assertRaises(ValueError): data_prep.make_features(sample_frame())

    def test_train_scaling_constant_column_and_invalid_input(self):
        tr, va, mean, scale = data_prep.standardize([[10, 1], [20, 1]], [[1000, 1]])
        np.testing.assert_allclose(va, [[197, 0]])
        np.testing.assert_allclose(mean, [15, 1]); np.testing.assert_allclose(scale, [5, 1])
        with self.assertRaises(ValueError): data_prep.standardize([[np.nan]])
        with self.assertRaises(ValueError): data_prep.standardize(np.empty((0, 2)))

    def test_training_tier_threshold_and_invalid_threshold(self):
        _, threshold = data_prep.make_tier_label([100, 200, 300])
        np.testing.assert_array_equal(data_prep.make_tier_label([200, 201, 10000], threshold)[0], [0, 1, 1])
        with self.assertRaises(ValueError): data_prep.make_tier_label([100], np.nan)

    def test_split_repeatability_disjoint_source_ids(self):
        frame = pd.concat([sample_frame()] * 10, ignore_index=True)
        frame["source_row_id"] = np.arange(len(frame))
        one, two = data_prep.split_dataframe(frame), data_prep.split_dataframe(frame)
        for left, right in zip(one, two):
            np.testing.assert_array_equal(left.source_row_id, right.source_row_id)
        ids = [set(part.source_row_id) for part in one]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])

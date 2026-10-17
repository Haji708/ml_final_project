"""Member 1 contracts: shared preparation and Task A regression only."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import data_prep
from src.linear_regression import LinearRegressionNormalEquation, RidgeRegressionNormalEquation
from src import member1_metrics as metrics
from src.member1_regression import (
    _diagnostic_cv, _validation_search, linear_feature_matrix,
    run_member1_experiments, run_member1_pipeline, validation_error_tables,
)


def raw_fixture(n=120):
    area = np.linspace(30, 180, n)
    rooms = 1 + np.arange(n) % 4
    return pd.DataFrame({
        "estate_id": np.arange(n), "price": 50000 + 1200 * area + 5000 * rooms,
        "Sahə": [f"{value} m²" for value in area], "Kateqoriya": "Yeni tikili",
        "Otaq sayı": rooms, "Mərtəbə": "3 / 12", "lat": 40.4, "lng": 49.9,
        "city": "Bakı", "location": np.where(np.arange(n) % 2, "north", "south"),
        "views": np.arange(n) + 10, "currency_x": "AZN", "currency_y": "AZN",
        "Torpaq sahəsi": None,
    })


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.directory.name) / "fixture.csv"
        raw_fixture().to_csv(self.csv_path, index=False)

    def tearDown(self):
        self.directory.cleanup()

    def test_preparation_import_does_not_import_models_or_sklearn(self):
        script = (
            "import sys; from src import data_prep; "
            "assert 'sklearn' not in sys.modules; "
            "assert 'src.member1_regression' not in sys.modules; "
            "assert 'src.linear_regression' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", script], cwd=data_prep.PROJECT_ROOT,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_default_paths_work_outside_project_directory(self):
        canonical = self.csv_path
        nested = Path(self.directory.name) / "nested" / "fixture.csv"
        with patch.object(data_prep, "_DATA_CANDIDATES", (nested, canonical)):
            self.assertEqual(data_prep.resolve_data_path(), canonical.resolve())

    def test_missing_path_explains_attempted_locations(self):
        with self.assertRaisesRegex(FileNotFoundError, "Pass its actual path"):
            data_prep.resolve_data_path(Path(self.directory.name) / "absent.csv")

    def test_shared_preparation_requires_validation_and_test_splits(self):
        for options in ({"val_size": 0}, {"test_size": 0}):
            with self.assertRaisesRegex(ValueError, "nonempty validation and test"):
                data_prep.prepare_data(self.csv_path, **options)

    def test_prepared_contract_shapes_and_training_threshold(self):
        prepared = data_prep.prepare_data(self.csv_path)
        self.assertEqual([len(prepared[f"y_{split}"]) for split in ("train", "validation", "test")], [84, 18, 18])
        threshold = float(np.median(prepared["y_train"]))
        self.assertEqual(prepared["tier_threshold"], threshold)
        for split in ("train", "validation", "test"):
            self.assertEqual(prepared[f"X_{split}"].shape, prepared[f"X_{split}_scaled"].shape)
            np.testing.assert_array_equal(prepared[f"y_tier_{split}"], (prepared[f"y_{split}"] > threshold).astype(int))
            self.assertTrue(np.isfinite(prepared[f"X_{split}_scaled"]).all())

    def test_rank_strata_are_proportionally_represented(self):
        frame = pd.DataFrame({"price": 10000 + np.arange(1000), "source_row_id": np.arange(1000)})
        train, validation, test = data_prep.split_dataframe(frame)
        for part, expected_per_band in ((train, 70), (validation, 15), (test, 15)):
            counts = (part.source_row_id // 100).value_counts().sort_index().to_numpy()
            np.testing.assert_array_equal(counts, np.full(10, expected_per_band))

    def test_split_is_repeatable_disjoint_and_covers_every_record(self):
        frame = data_prep.clean(raw_fixture())
        first, second = data_prep.split_dataframe(frame), data_prep.split_dataframe(frame)
        ids = []
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left.source_row_id, right.source_row_id)
            ids.append(set(left.source_row_id))
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertEqual(set.union(*ids), set(frame.source_row_id))

    def test_compact_metadata_uses_counts_and_hashes(self):
        prepared = data_prep.prepare_data(self.csv_path)
        metadata = prepared["metadata"]
        self.assertNotIn("split_row_ids", metadata)
        self.assertEqual(len(metadata["dataset"]["sha256"]), 64)
        for name, details in metadata["splits"].items():
            self.assertEqual(details["rows"], len(prepared[f"y_{name}"]))
            self.assertEqual(len(details["source_rows_sha256"]), 64)
            self.assertEqual(details["standard_rows"] + details["premium_rows"], details["rows"])
        self.assertLess(len(json.dumps(metadata)), 16000)
        repeated = data_prep.prepare_data(self.csv_path)
        self.assertEqual(metadata["splits"], repeated["metadata"]["splits"])

    def test_both_area_fields_convert_sot_and_hectares(self):
        raw = raw_fixture(3)
        raw["Kateqoriya"] = "Torpaq"
        raw["Sahə"] = ["2 sot", "1 ha", "80 m²"]
        raw["Torpaq sahəsi"] = ["2 sot", "1 hektar", "80 m²"]
        clean = data_prep.clean(raw)
        processor = data_prep.TabularPreprocessor()
        X, _ = processor.fit_transform(clean)
        for name in ("Sahə_m2", "Torpaq_sahəsi_m2"):
            np.testing.assert_allclose(X[:, processor.feature_names_.index(name)], [200, 10000, 80])

    def test_missing_land_has_explicit_indicator(self):
        frame = data_prep.clean(raw_fixture(4))
        frame.loc[1, "Torpaq sahəsi"] = "2 sot"
        processor = data_prep.TabularPreprocessor()
        X, _ = processor.fit_transform(frame)
        np.testing.assert_array_equal(X[:, processor.feature_names_.index("Torpaq_missing")], [1, 0, 1, 1])

    def test_leakage_and_identifiers_cannot_be_features(self):
        raw = raw_fixture(12)
        raw["total_price"] = raw.price
        raw["unit_price"] = 1234
        raw["owner_name"] = "example"
        frame = data_prep.clean(raw)
        processor = data_prep.TabularPreprocessor().fit(frame)
        forbidden = {"price", "unit_price", "total_price", "estate_id", "source_row_id", "owner_name"}
        self.assertFalse(forbidden & set(processor.feature_names_))
        self.assertFalse({"total_price", "unit_price", "owner_name"} & set(frame.columns))

    def test_currency_is_checked_not_silently_relabelled(self):
        raw = raw_fixture(4)
        raw.loc[0, "currency_x"] = "USD"
        with self.assertRaisesRegex(ValueError, "conversion policy"):
            data_prep.clean(raw)

    def test_duplicate_id_uses_latest_timestamp(self):
        raw = raw_fixture(2)
        raw["estate_id"] = "same"
        raw["datetime_scrape_y"] = ["2024-11-10", "2024-10-10"]
        self.assertEqual(data_prep.clean(raw).price.tolist(), [raw.iloc[0].price])

    def test_validation_cannot_change_training_imputation(self):
        frame = data_prep.clean(raw_fixture(6))
        processor = data_prep.TabularPreprocessor().fit(frame.iloc[:3])
        before = dict(processor.numeric_medians_)
        held_out = frame.iloc[3:].copy()
        held_out["Sahə"] = [None, "9999999 m²", "8888888 m²"]
        X, _ = processor.transform(held_out)
        self.assertEqual(processor.numeric_medians_, before)
        self.assertEqual(X[0, processor.feature_names_.index("Sahə_m2")], before["Sahə_m2"])

    def test_unknown_categories_preserve_feature_columns(self):
        frame = data_prep.clean(raw_fixture(6))
        processor = data_prep.TabularPreprocessor().fit(frame.iloc[:3])
        held_out = frame.iloc[3:].copy()
        held_out["location"] = "unseen location"
        X, _ = processor.transform(held_out)
        positions = [i for i, name in enumerate(processor.feature_names_) if name.startswith("location=")]
        self.assertEqual(X.shape[1], len(processor.feature_names_))
        self.assertTrue((X[:, positions] == 0).all())

    def test_train_only_scaling_handles_constant_columns(self):
        train, held_out, mean, scale = data_prep.standardize([[10, 1], [20, 1]], [[1000, 1]])
        np.testing.assert_allclose(train.mean(axis=0), [0, 0])
        np.testing.assert_allclose(mean, [15, 1])
        np.testing.assert_allclose(scale, [5, 1])
        np.testing.assert_allclose(held_out, [[197, 0]])

    def test_preprocessing_requires_fit_and_helper_requires_state(self):
        frame = data_prep.clean(raw_fixture(4))
        with self.assertRaises(RuntimeError):
            data_prep.TabularPreprocessor().transform(frame)
        with self.assertRaises(ValueError):
            data_prep.make_features(frame)


class ModelAndMetricTests(unittest.TestCase):
    def test_known_linear_coefficients(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 2))
        y = 5 + 2 * X[:, 0] - 3 * X[:, 1]
        model = LinearRegressionNormalEquation().fit(X, y)
        np.testing.assert_allclose(model.coef_, [2, -3], atol=1e-10)
        self.assertAlmostEqual(model.intercept_, 5)

    def test_linear_and_ridge_match_sklearn(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(80, 4))
        y = 123 + X @ np.array([2, -3, 1, 0.5]) + rng.normal(size=80)
        for custom, reference in ((LinearRegressionNormalEquation(), LinearRegression()),
                                  (RidgeRegressionNormalEquation(2), Ridge(alpha=2))):
            custom.fit(X, y); reference.fit(X, y)
            np.testing.assert_allclose(custom.predict(X), reference.predict(X), atol=1e-9)

    def test_singular_and_badly_scaled_designs_remain_accurate(self):
        for X in (np.column_stack([np.arange(12), np.arange(12), np.ones(12)]),
                  np.random.default_rng(1).normal(size=(100, 2)) * [1, 1e8]):
            y = 3 + 2 * X[:, 0] + 1e-8 * X[:, 1]
            model = LinearRegressionNormalEquation().fit(X, y)
            np.testing.assert_allclose(model.predict(X), y, atol=1e-6)

    def test_ridge_does_not_penalize_intercept(self):
        model = RidgeRegressionNormalEquation(1e6).fit(np.ones((10, 3)), np.full(10, 12345))
        self.assertAlmostEqual(model.intercept_, 12345)
        np.testing.assert_allclose(model.predict(np.ones((2, 3))), [12345, 12345])

    def test_invalid_alpha_and_unfitted_prediction_are_rejected(self):
        for alpha in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                RidgeRegressionNormalEquation(alpha)
        with self.assertRaises(RuntimeError):
            LinearRegressionNormalEquation().predict([[1, 2]])

    def test_regression_metrics_match_known_values_and_sklearn(self):
        actual = np.array([1, 2, 3, 10], dtype=float)
        prediction = np.array([2, 2, 2, 8], dtype=float)
        self.assertAlmostEqual(metrics.rmse(actual, prediction), np.sqrt(mean_squared_error(actual, prediction)))
        self.assertAlmostEqual(metrics.mae(actual, prediction), mean_absolute_error(actual, prediction))
        self.assertAlmostEqual(metrics.r2(actual, prediction), r2_score(actual, prediction))
        self.assertEqual(metrics.regression_scores(actual, actual), {"RMSE": 0, "MAE": 0, "R2": 1})

    def test_metric_edge_cases_are_explicit(self):
        self.assertTrue(np.isnan(metrics.r2([1], [1])))
        self.assertEqual(metrics.r2([5, 5], [5, 5]), 1)
        self.assertEqual(metrics.r2([5, 5], [4, 4]), 0)
        for actual, prediction in (([], []), ([1], [1, 2]), ([1], [np.inf]), ([[1, 2]], [[1, 2]])):
            with self.assertRaises(ValueError):
                metrics.rmse(actual, prediction)

    def test_linear_log_transform_preserves_shared_input(self):
        original = np.array([[200, 0, 100, 1]], dtype=float)
        logged = linear_feature_matrix(original, ["Sahə_m2", "Torpaq_sahəsi_m2", "views", "flag"], "log_size")
        np.testing.assert_array_equal(original, [[200, 0, 100, 1]])
        np.testing.assert_allclose(logged, [[np.log1p(200), 0, np.log1p(100), 1]])

    def test_interactions_fit_different_area_slopes_by_property_type(self):
        area = np.tile(np.linspace(10, 100, 20), 2)
        category = np.repeat([0, 1], 20)
        X = np.column_stack([area, category == 0, category == 1])
        names = ["Sahə_m2", "Kateqoriya=Apartment", "Kateqoriya=Land"]
        y = 50_000 + area * np.where(category == 0, 2_000, 100)
        before = X.copy()
        transformed = linear_feature_matrix(X, names, "raw_interactions")
        model = LinearRegressionNormalEquation().fit(transformed, y)
        unseen = np.array([[55, 1, 0], [55, 0, 1]])
        prediction = model.predict(linear_feature_matrix(unseen, names, "raw_interactions"))
        np.testing.assert_allclose(prediction, [160_000, 55_500], atol=1e-6)
        np.testing.assert_array_equal(X, before)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.csv = self.root / "fixture.csv"
        raw_fixture().to_csv(self.csv, index=False)
        self.prepared = data_prep.prepare_data(self.csv)

    def tearDown(self):
        self.directory.cleanup()

    def test_validation_selection_never_needs_test_arrays(self):
        train_validation_only = {key: value for key, value in self.prepared.items()
                                 if key not in ("X_test", "X_test_scaled", "y_test", "y_tier_test", "test_frame")}
        choices, trials, _ = _validation_search(train_validation_only, True)
        self.assertEqual(set(choices), {"Linear", "Ridge", "Tree"})
        self.assertEqual(len(trials), 70)

    def test_cv_refits_preprocessing_for_every_fold(self):
        seen = []
        original = data_prep.TabularPreprocessor.fit
        def recording(processor, frame):
            seen.append(set(frame.source_row_id))
            return original(processor, frame)
        choice = {"Ridge": {"alpha": 1, "feature_mode": "log_size", "target_mode": "price"}}
        with patch.object(data_prep.TabularPreprocessor, "fit", new=recording):
            records = _diagnostic_cv(self.prepared["train_frame"], choice, 3, 42)
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(len(rows) == 56 for rows in seen))
        self.assertEqual(len(records), 6)

    def test_runner_preserves_shared_data_and_writes_compact_results(self):
        arrays_before = {name: value.copy() for name, value in self.prepared.items() if isinstance(value, np.ndarray)}
        metadata_before = json.dumps(self.prepared["metadata"], sort_keys=True)
        output = self.root / "results"
        result = run_member1_experiments(self.prepared, output, cv_folds=0)
        for name, expected in arrays_before.items():
            np.testing.assert_array_equal(self.prepared[name], expected)
        self.assertEqual(json.dumps(self.prepared["metadata"], sort_keys=True), metadata_before)
        self.assertEqual({path.name for path in output.iterdir()}, {
            "metadata.json", "test_metrics.csv", "validation_trials.csv", "cv_folds.csv", "cv_summary.csv",
            "validation_metrics.csv", "validation_errors.csv", "validation_worst_errors.csv",
            "validation_error_by_category.csv"})
        self.assertEqual(len(result["test_metrics"]), 6)
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        self.assertNotIn("split_row_ids", metadata)
        self.assertLess((output / "metadata.json").stat().st_size, 16000)
        self.assertIn("Tree", metadata["selected_configs"])
        self.assertTrue(all(value < 1e-4 for value in result["prediction_differences"].values()))
        self.assertEqual(pd.read_csv(output / "cv_summary.csv").shape[0], 0)

    def test_validation_diagnostics_keep_original_row_identity(self):
        truth = self.prepared["y_validation"]
        prediction = truth.copy()
        prediction[2] += 10_000
        prediction[0] += 100
        errors, worst, categories = validation_error_tables(
            self.prepared, {"Custom Ridge": prediction}, top_n=2)
        self.assertEqual(len(errors), len(truth))
        self.assertEqual(worst["source_row_id"].tolist(),
                         self.prepared["validation_frame"]["source_row_id"].iloc[[2, 0]].tolist())
        np.testing.assert_allclose(worst["absolute_error_AZN"], [10_000, 100])
        self.assertEqual(categories["rows"].sum(), len(truth))
        self.assertTrue(set(errors.source_row_id).isdisjoint(self.prepared["test_frame"].source_row_id))

    def test_log_target_predictions_are_reported_in_azn(self):
        frame = raw_fixture()
        area = frame["Sahə"].map(data_prep._extract_number).to_numpy()
        frame["price"] = np.expm1(8 + 0.015 * area)
        frame.to_csv(self.csv, index=False)
        prepared = data_prep.prepare_data(self.csv)
        with patch("builtins.print"):
            result = run_member1_experiments(prepared, self.root / "logged", cv_folds=0,
                                             include_tree_baseline=False)
        self.assertEqual(result["selections"]["Linear"]["target_mode"], "log_price")
        np.testing.assert_allclose(result["test_predictions"]["Custom Linear"],
                                   prepared["y_test"], rtol=1e-9)
        self.assertGreater(result["validation_results"]["Custom Linear"]["R2"], 0.999999)

    def test_compatibility_wrapper_preserves_previous_returned_keys(self):
        with patch.object(data_prep, "prepare_data", return_value=self.prepared) as preparing:
            result = run_member1_pipeline(self.csv, self.root / "compatibility", cv_folds=0)
        preparing.assert_called_once_with(self.csv)
        for split in ("train", "validation", "test"):
            for name in (f"X_{split}", f"X_{split}_scaled", f"y_{split}", f"y_tier_{split}"):
                self.assertIn(name, result)


if __name__ == "__main__":
    unittest.main()

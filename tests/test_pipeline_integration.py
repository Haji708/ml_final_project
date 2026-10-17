"""Integration checks for selected regression and SVM configurations."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import data_prep, run_all
from src.member1_metrics import regression_scores


def fixture_frame(n=120):
    rng = np.random.default_rng(17)
    area = rng.uniform(30, 180, n)
    land = np.arange(n) % 3 == 0
    return pd.DataFrame({
        "estate_id": np.arange(n),
        "price": 40_000 + area * np.where(land, 300, 1_800),
        "Sahə": [f"{a} m²" for a in area],
        "Kateqoriya": np.where(land, "Torpaq", "Yeni tikili"),
        "Otaq sayı": 1 + np.arange(n) % 4, "Mərtəbə": "3 / 12",
        "lat": 40.4, "lng": 49.9, "city": "Bakı", "location": "Yasamal",
        "views": np.arange(n) + 10, "Torpaq sahəsi": None,
        "currency_x": "AZN", "currency_y": "AZN",
    })


class IntegrationTests(unittest.TestCase):
    def test_module_import_keeps_scratch_tree_enabled(self):
        result = subprocess.run(
            [sys.executable, "-c", "from src.run_all import DecisionTree; assert DecisionTree is not None"],
            cwd=data_prep.PROJECT_ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_svm_search_selects_reproducible_family_configurations(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(60, 3))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        _, trials = run_all.select_svm_config(
            X[:40], y[:40], X[40:], y[40:], iteration_budgets=(50, 120),
            component_counts=(8, 16), gammas=(0.005, 0.5), random_state=17)
        self.assertEqual(set(trials.feature_map), {"linear", "rff"})
        self.assertEqual(set(trials.n_iters), {50, 120})
        self.assertIn(16, trials.loc[trials.feature_map == "rff", "n_components"].values)
        self.assertTrue(np.isfinite(trials["final_objective"]).all())
        for _, family in trials.groupby("feature_map"):
            winner = family.sort_values(["F1", "AUC"], ascending=False, kind="stable").iloc[0]
            model = run_all.svm_from_trial(winner, 17).fit(X[:40], y[:40])
            _, _, f1 = run_all.evaluate.precision_recall_f1(y[40:], model.predict(X[40:]))
            self.assertAlmostEqual(f1, winner.F1)

    def test_team_runner_uses_member1_predictions_and_selected_rff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = root / "fixture.csv"
            fixture_frame().to_csv(csv, index=False)
            captured = {}
            original = run_all.member1_regression.run_member1_experiments

            def recording(*args, **kwargs):
                result = original(*args, **kwargs)
                captured["member1"] = result
                return result

            with patch.object(run_all.member1_regression, "run_member1_experiments", side_effect=recording), \
                 patch.object(data_prep, "prepare_data", wraps=data_prep.prepare_data) as preparing, \
                 patch.object(run_all, "plot_svm_weights"), \
                 patch.object(run_all, "plot_tree_feature_importances"), \
                 patch.object(run_all, "plot_tree_validation_curves"), \
                 patch.object(run_all, "plot_error_analysis"), redirect_stdout(StringIO()):
                run_all.main(["--data", str(csv), "--output", str(root / "report"),
                              "--cv-folds", "0", "--seed", "17",
                              "--svm-iterations", "50", "120", "--svm-components", "8", "16"])
            preparing.assert_called_once()
            summary = pd.read_csv(root / "report" / "validation_summary.csv", index_col=0)
            result = captured["member1"]
            for source, display in (("Custom Linear", "Linear (Scratch)"),
                                    ("Custom Ridge", "Ridge (Scratch)")):
                for metric, value in result["validation_results"][source].items():
                    self.assertAlmostEqual(summary.loc[display, metric], value)
                recomputed = regression_scores(result["y_test"], result["test_predictions"][source])
                self.assertEqual(recomputed, result["regression_results"][source])
            configs = json.loads((root / "report" / "selected_configs.json").read_text())
            trials = pd.read_csv(root / "report" / "svm_validation_trials.csv")
            rff = trials[trials.feature_map == "rff"].sort_values(
                ["F1", "AUC"], ascending=False, kind="stable").iloc[0]
            self.assertEqual(configs["regression"], result["selections"])
            for key in ("gamma", "lambda_", "n_iters", "n_components"):
                self.assertEqual(configs["svm_rff"][key], rff[key])
            self.assertAlmostEqual(summary.loc["PegasosSVM (RFF)", "F1"], rff.F1)
            convergence = pd.read_csv(root / "report" / "svm_convergence.csv")
            self.assertEqual(set(convergence["model"]), {"PegasosSVM (Scratch)", "PegasosSVM (RFF)"})


if __name__ == "__main__":
    unittest.main()

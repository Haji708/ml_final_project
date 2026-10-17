"""Member 1's Task A experiments; shared preparation lives in data_prep.py.

No classification metrics, EDA, custom decision tree, SVM, or final team runner
are implemented here. The sklearn tree is a Task A comparison baseline only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

from . import data_prep
from .linear_regression import LinearRegressionNormalEquation, RidgeRegressionNormalEquation
from .member1_metrics import regression_scores


FEATURE_MODES = ("raw", "log_size", "raw_interactions", "log_size_interactions")
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)


def regression_metrics(y_true, y_pred):
    """Backward-compatible name for the regression-only metric wrapper."""
    return regression_scores(y_true, y_pred)


def linear_feature_matrix(X, feature_names, feature_mode):
    """Copy features, optionally logging sizes and adding area/category terms.

    Category columns come from the training-fitted shared preprocessor. No
    category vocabulary or statistics are learned from validation/test rows.
    """
    result = np.asarray(X, dtype=float).copy()
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {FEATURE_MODES}")
    if feature_mode.startswith("log_size"):
        for name in ("Sahə_m2", "Torpaq_sahəsi_m2", "views"):
            if name in feature_names:
                position = feature_names.index(name)
                if np.any(result[:, position] < 0):
                    raise ValueError(f"{name} must be nonnegative for log1p")
                result[:, position] = np.log1p(result[:, position])
    if feature_mode.endswith("_interactions"):
        categories = [i for i, name in enumerate(feature_names)
                      if name.startswith("Kateqoriya=")]
        interactions = [result[:, feature_names.index(area), None] * result[:, categories]
                        for area in ("Sahə_m2", "Torpaq_sahəsi_m2")
                        if area in feature_names and categories]
        if interactions:
            result = np.column_stack([result, *interactions])
    return result


def target_values(y, mode):
    """Prepare price or log1p(price) training targets."""
    values = np.asarray(y, dtype=float)
    if mode == "price":
        return values
    if mode == "log_price":
        if np.any(values < 0):
            raise ValueError("Prices must be nonnegative for log1p")
        return np.log1p(values)
    raise ValueError("target_mode must be price or log_price")


def price_predictions(model, X, mode):
    """Return finite AZN predictions, with the same nonnegative rule for all models."""
    prediction = model.predict(X)
    with np.errstate(over="ignore", invalid="ignore"):
        if mode == "log_price":
            prediction = np.expm1(prediction)
        elif mode != "price":
            raise ValueError("target_mode must be price or log_price")
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Predictions overflowed when converting back to AZN")
    return np.maximum(prediction, 0.0)


def _models(family, choice, seed):
    if family == "Linear":
        return {"Custom Linear": LinearRegressionNormalEquation(),
                "sklearn Linear": LinearRegression()}
    if family == "Ridge":
        return {"Custom Ridge": RidgeRegressionNormalEquation(choice["alpha"]),
                "sklearn Ridge": Ridge(alpha=choice["alpha"], solver="svd")}
    if family == "Tree":
        return {"sklearn DecisionTreeRegressor": DecisionTreeRegressor(
            max_depth=choice["max_depth"], min_samples_leaf=choice["min_samples_leaf"],
            criterion="squared_error", random_state=seed)}
    raise ValueError(f"Unknown regression family: {family}")


def _validation_search(prepared, include_tree_baseline):
    """Choose settings using TRAIN and VALIDATION only, never test outcomes."""
    y_train, y_validation = prepared["y_train"], prepared["y_validation"]
    trials, selections, cache = [], {}, {}
    for feature_mode in FEATURE_MODES:
        train = linear_feature_matrix(prepared["X_train"], prepared["feature_names"], feature_mode)
        validation = linear_feature_matrix(prepared["X_validation"], prepared["feature_names"], feature_mode)
        cache[feature_mode] = data_prep.standardize(train, validation)
        scaled_train, scaled_validation = cache[feature_mode][:2]
        for target_mode in ("price", "log_price"):
            for family, alphas in (("Linear", (0.0,)), ("Ridge", RIDGE_ALPHAS)):
                for alpha in alphas:
                    choice = {"family": family, "alpha": alpha, "feature_mode": feature_mode,
                              "target_mode": target_mode}
                    model = next(iter(_models(family, choice, data_prep.SEED).values()))
                    model.fit(scaled_train, target_values(y_train, target_mode))
                    try:
                        val_rmse = regression_scores(y_validation, price_predictions(model, scaled_validation, target_mode))["RMSE"]
                        train_rmse = regression_scores(y_train, price_predictions(model, scaled_train, target_mode))["RMSE"]
                    except FloatingPointError:
                        val_rmse = train_rmse = float("inf")
                    trial = {**choice, "validation_RMSE": val_rmse, "training_RMSE": train_rmse}
                    trials.append(trial)
                    if np.isfinite(val_rmse) and (family not in selections or val_rmse < selections[family]["validation_RMSE"]):
                        selections[family] = trial.copy()

    if include_tree_baseline:
        seed = prepared["metadata"]["seed"]
        for depth in (4, 8, 12):
            for min_leaf in (5, 20):
                choice = {"family": "Tree", "max_depth": depth, "min_samples_leaf": min_leaf,
                          "feature_mode": "raw_unscaled", "target_mode": "price"}
                model = next(iter(_models("Tree", choice, seed).values()))
                model.fit(prepared["X_train"], y_train)
                val_rmse = regression_scores(y_validation, model.predict(prepared["X_validation"]))["RMSE"]
                train_rmse = regression_scores(y_train, model.predict(prepared["X_train"]))["RMSE"]
                trial = {**choice, "validation_RMSE": val_rmse, "training_RMSE": train_rmse}
                trials.append(trial)
                if "Tree" not in selections or val_rmse < selections["Tree"]["validation_RMSE"]:
                    selections["Tree"] = trial.copy()
    if "Linear" not in selections or "Ridge" not in selections:
        raise RuntimeError("No finite Linear/Ridge configuration was found")
    return deepcopy(selections), trials, cache


def _diagnostic_cv(train_df, selections, folds, seed):
    """Fixed-config training-only CV; preprocessing is fitted anew per fold.

    Configurations were already selected on overlapping data. These are
    descriptive variability checks, not nested-CV estimates of the full search.
    """
    columns = ["Model", "fold", "RMSE", "MAE", "R2"]
    if folds == 0:
        return pd.DataFrame(columns=columns)
    if folds < 2 or len(train_df) < 2 * folds:
        raise ValueError("CV requires at least two folds and two held-out rows per fold")
    partitions = np.array_split(np.random.default_rng(seed).permutation(len(train_df)), folds)
    records = []
    for fold, held_out in enumerate(partitions, start=1):
        fitting = np.concatenate([part for index, part in enumerate(partitions, start=1) if index != fold])
        processor = data_prep.TabularPreprocessor()
        X_train, y_train = processor.fit_transform(train_df.iloc[fitting])
        X_validation, y_validation = processor.transform(train_df.iloc[held_out])
        cache = {}
        for family, choice in selections.items():
            if family == "Tree":
                train, validation = X_train, X_validation
            else:
                mode = choice["feature_mode"]
                if mode not in cache:
                    train = linear_feature_matrix(X_train, processor.feature_names_, mode)
                    validation = linear_feature_matrix(X_validation, processor.feature_names_, mode)
                    cache[mode] = data_prep.standardize(train, validation)[:2]
                train, validation = cache[mode]
            for name, model in _models(family, choice, seed).items():
                model.fit(train, target_values(y_train, choice["target_mode"]))
                prediction = price_predictions(model, validation, choice["target_mode"])
                records.append({"Model": name, "fold": fold, **regression_scores(y_validation, prediction)})
    return pd.DataFrame(records, columns=columns)


def _print_results(frame):
    print("\nMember 1 Task A holdout results (AZN)")
    print(frame[["Model", "RMSE", "MAE", "R2"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def validation_error_tables(prepared, predictions, top_n=50):
    """Return row errors, worst rows per model, and category-level scores.

    These are validation diagnostics only; no rows are removed or clipped.
    Source-row IDs preserve the association with the original CSV.
    """
    source = prepared["validation_frame"].reset_index(drop=True)
    context = [column for column in ("source_row_id", "Kateqoriya", "Sahə",
               "Torpaq sahəsi", "Otaq sayı", "city", "location") if column in source]
    base = source[context].copy()
    for name in ("Sahə_m2", "Torpaq_sahəsi_m2"):
        if name in prepared["feature_names"]:
            base[name] = prepared["X_validation"][:, prepared["feature_names"].index(name)]
    base["actual_price_AZN"] = prepared["y_validation"]
    tables, groups = [], []
    for model_name, prediction in predictions.items():
        errors = base.copy()
        errors.insert(0, "Model", model_name)
        errors["predicted_price_AZN"] = prediction
        errors["residual_AZN"] = errors["actual_price_AZN"] - prediction
        errors["absolute_error_AZN"] = np.abs(errors["residual_AZN"])
        tables.append(errors.sort_values("absolute_error_AZN", ascending=False, kind="stable"))
        if "Kateqoriya" in errors:
            for category, rows in errors.groupby("Kateqoriya", dropna=False, sort=False):
                groups.append({"Model": model_name, "Kateqoriya": category,
                               "rows": len(rows), **regression_scores(
                                   rows["actual_price_AZN"].to_numpy(),
                                   rows["predicted_price_AZN"].to_numpy())})
    all_errors = pd.concat(tables, ignore_index=True)
    worst = all_errors.groupby("Model", sort=False).head(top_n).reset_index(drop=True)
    return all_errors, worst, pd.DataFrame(groups)


def run_member1_experiments(
    prepared: dict,
    output_dir: str | Path | None = None,
    cv_folds: int = 5,
    *,
    include_tree_baseline: bool = True,
) -> dict:
    """Consume the shared prepared data. Do not reload it or fit shared preprocessing.

    Returns the supplied prepared-data entries plus Member 1 results/models.
    No shared feature array is modified. Member 4 can call this after calling
    prepare_data once, then pass the same prepared inputs to other members.
    """
    if not isinstance(cv_folds, int) or cv_folds not in (0,) and cv_folds < 2:
        raise ValueError("cv_folds must be 0 or an integer of at least 2")
    needed = ("X_train", "X_validation", "X_test", "y_train", "y_validation", "y_test",
              "feature_names", "train_frame", "metadata")
    missing = [name for name in needed if name not in prepared]
    if missing:
        raise ValueError(f"Call data_prep.prepare_data first; missing keys: {missing}")
    selections, trials, cache = _validation_search(prepared, include_tree_baseline)
    # All selections are now frozen. Test outcomes are used only below.
    results, fitted_models, predictions, rows, differences = {}, {}, {}, [], {}
    validation_results, validation_predictions, validation_rows = {}, {}, []
    for family, choice in selections.items():
        if family == "Tree":
            train, validation, test = (prepared["X_train"], prepared["X_validation"], prepared["X_test"])
        else:
            train, validation, mean, scale = cache[choice["feature_mode"]]
            test = (linear_feature_matrix(prepared["X_test"], prepared["feature_names"], choice["feature_mode"]) - mean) / scale
        for name, model in _models(family, choice, prepared["metadata"]["seed"]).items():
            start = perf_counter()
            model.fit(train, target_values(prepared["y_train"], choice["target_mode"]))
            fit_seconds = perf_counter() - start
            start = perf_counter()
            prediction = price_predictions(model, test, choice["target_mode"])
            predict_seconds = perf_counter() - start
            val_prediction = price_predictions(model, validation, choice["target_mode"])
            validation_predictions[name] = val_prediction
            validation_results[name] = regression_scores(prepared["y_validation"], val_prediction)
            validation_rows.append({"Model": name, **validation_results[name], **choice})
            scores = regression_scores(prepared["y_test"], prediction)
            results[name], fitted_models[name], predictions[name] = scores, model, prediction
            rows.append({"Model": name, **scores, **choice,
                         "fit_seconds": fit_seconds, "predict_seconds": predict_seconds})
        print(f"Selected {family}: {choice}")
    median_prediction = np.full(len(prepared["y_test"]), np.median(prepared["y_train"]))
    results["Training median"] = regression_scores(prepared["y_test"], median_prediction)
    rows.append({"Model": "Training median", **results["Training median"], "fit_seconds": 0.0})
    validation_results["Training median"] = regression_scores(
        prepared["y_validation"], np.full(len(prepared["y_validation"]), np.median(prepared["y_train"])))
    validation_rows.append({"Model": "Training median", **validation_results["Training median"]})
    for family in ("Linear", "Ridge"):
        differences[family] = float(np.max(np.abs(predictions[f"Custom {family}"] - predictions[f"sklearn {family}"])))
        print(f"Max Custom/sklearn {family} prediction difference: {differences[family]:.6g} AZN")

    cv = _diagnostic_cv(prepared["train_frame"], selections, cv_folds, prepared["metadata"]["seed"])
    if cv.empty:
        summary = pd.DataFrame(columns=["Model", "RMSE_mean", "RMSE_std", "MAE_mean", "MAE_std", "R2_mean", "R2_std"])
    else:
        summary = cv.groupby("Model", sort=True)[["RMSE", "MAE", "R2"]].agg(["mean", "std"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = summary.reset_index()

    output = Path(output_dir) if output_dir is not None else data_prep.PROJECT_ROOT / "report" / "member1_checked"
    output.mkdir(parents=True, exist_ok=True)
    test_frame, trial_frame = pd.DataFrame(rows), pd.DataFrame(trials)
    test_frame.to_csv(output / "test_metrics.csv", index=False)
    trial_frame.to_csv(output / "validation_trials.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(output / "validation_metrics.csv", index=False)
    diagnostic_predictions = {name: prediction for name, prediction in validation_predictions.items()
                              if name.startswith("Custom ") or name.startswith("sklearn DecisionTree")}
    errors, worst, by_category = validation_error_tables(prepared, diagnostic_predictions)
    errors.to_csv(output / "validation_errors.csv", index=False)
    worst.to_csv(output / "validation_worst_errors.csv", index=False)
    by_category.to_csv(output / "validation_error_by_category.csv", index=False)
    cv.to_csv(output / "cv_folds.csv", index=False)
    summary.to_csv(output / "cv_summary.csv", index=False)
    metadata = deepcopy(prepared["metadata"])
    metadata.update({
        "selected_configs": selections,
        "search_feature_modes": list(FEATURE_MODES),
        "search_ridge_alphas": list(RIDGE_ALPHAS),
        "selection_metric": "validation RMSE in AZN after inverse target transformation",
        "diagnostic_split": "validation only; top 50 absolute errors per custom model and tree baseline",
        "custom_sklearn_max_difference_AZN": differences,
        "cv_folds": cv_folds,
        "cv_note": "Training-only fixed-config descriptive CV; not nested CV and not test-set confidence intervals.",
        "prediction_rule": "Invert log1p with expm1 where selected; clip all AZN predictions below zero to zero.",
        "completed_scope": "Member 1 shared preparation, Linear/Ridge, and sklearn Task A tree baseline only.",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "member1_source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ("data_prep.py", "linear_regression.py", "member1_metrics.py", "member1_regression.py")},
    })
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _print_results(test_frame)
    print(f"Results saved to {output.resolve()}")
    print(f"Compact metadata: {(output / 'metadata.json').stat().st_size:,} bytes")
    return {**prepared, "regression_results": results, "fitted_models": fitted_models,
            "selections": selections, "best_ridge_alpha": selections["Ridge"]["alpha"],
            "prediction_differences": differences, "test_metrics": test_frame,
            "validation_results": validation_results, "validation_predictions": validation_predictions,
            "test_predictions": predictions,
            "validation_trials": trial_frame, "cv_results": cv, "cv_summary": summary,
            "output_dir": output, "experiment_metadata": metadata}


def run_member1_pipeline(data_path=None, output_dir=None, cv_folds=5):
    """Compatibility wrapper for the previous run_all.py; teammates use prepare_data."""
    return run_member1_experiments(data_prep.prepare_data(data_path), output_dir, cv_folds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run only Member 1's regression experiments")
    parser.add_argument("--data", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=data_prep.SEED)
    parser.add_argument("--no-tree-baseline", action="store_true", help="Run Linear/Ridge only")
    args = parser.parse_args()
    prepared = data_prep.prepare_data(args.data, seed=args.seed)
    metadata = prepared["metadata"]
    print(f"Rows raw/clean: {metadata['cleaning_audit']['raw_rows']}/{metadata['cleaning_audit']['clean_rows']}")
    print("Shared split sizes:", {name: values["rows"] for name, values in metadata["splits"].items()})
    print(f"Features: {metadata['feature_count']}; training tier threshold: {prepared['tier_threshold']:.2f} AZN")
    run_member1_experiments(prepared, args.output, args.cv_folds,
                            include_tree_baseline=not args.no_tree_baseline)


if __name__ == "__main__":
    main()

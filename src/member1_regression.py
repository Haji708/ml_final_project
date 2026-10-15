"""Checked Member 1 Linear/Ridge pipeline. Required tree/SVM integration is still separate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from . import data_prep, evaluate
from .linear_regression import LinearRegressionNormalEquation, RidgeRegressionNormalEquation


def regression_metrics(y_true, y_pred):
    return {"RMSE": evaluate.rmse(y_true, y_pred), "MAE": evaluate.mae(y_true, y_pred),
            "R2": evaluate.r2(y_true, y_pred)}


def linear_feature_matrix(X, feature_names, feature_mode):
    """Log sizes for LINEAR models only; keep original tree/SVM arrays."""
    result = np.asarray(X, dtype=float).copy()
    if feature_mode == "raw":
        return result
    if feature_mode != "log_size":
        raise ValueError("feature_mode must be raw or log_size")
    for column in ("Sahə_m2", "Torpaq_sahəsi_m2", "views"):
        if column in feature_names:
            j = feature_names.index(column)
            if np.any(result[:, j] < 0):
                raise ValueError(f"{column} must be nonnegative")
            result[:, j] = np.log1p(result[:, j])
    return result


def target_values(y, mode):
    if mode == "price":
        return np.asarray(y, dtype=float)
    if mode == "log_price":
        return np.log1p(y)
    raise ValueError("Target mode must be price or log_price")


def price_predictions(model, X, mode):
    prediction = model.predict(X)
    with np.errstate(over="ignore", invalid="ignore"):
        prediction = np.expm1(prediction) if mode == "log_price" else prediction
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite predictions after target inversion")
    # Preserve the author's stated nonnegative-price prediction rule.
    return np.maximum(prediction, 0.0)


def _custom_model(family, alpha):
    return LinearRegressionNormalEquation() if family == "Linear" else RidgeRegressionNormalEquation(alpha)


def _diagnostic_cv(train_df, selections, folds, seed):
    """Fixed-config training-only CV, with new preprocessing for every fold.

    These are descriptive variability estimates, not nested-CV estimates of
    the entire tuning procedure, because selections used overlapping data.
    """
    columns = ["Model", "fold", "RMSE", "MAE", "R2"]
    if folds == 0:
        return pd.DataFrame(columns=columns)
    if folds < 2 or len(train_df) < 2 * folds:
        raise ValueError("CV needs >=2 folds and >=2 held-out rows per fold")
    partitions = np.array_split(np.random.default_rng(seed).permutation(len(train_df)), folds)
    records = []
    for fold, held_out in enumerate(partitions):
        train_i = np.concatenate([p for j, p in enumerate(partitions) if j != fold])
        fitting, validation = train_df.iloc[train_i], train_df.iloc[held_out]
        processor = data_prep.TabularPreprocessor()
        Xtr, ytr = processor.fit_transform(fitting)
        Xva, yva = processor.transform(validation)
        for family, choice in selections.items():
            Atr = linear_feature_matrix(Xtr, processor.feature_names_, choice["feature_mode"])
            Ava = linear_feature_matrix(Xva, processor.feature_names_, choice["feature_mode"])
            Atr, Ava, _, _ = data_prep.standardize(Atr, Ava)
            models = {f"Custom {family}": _custom_model(family, choice["alpha"]),
                      f"sklearn {family}": LinearRegression() if family == "Linear" else Ridge(alpha=choice["alpha"], solver="svd")}
            for name, model in models.items():
                model.fit(Atr, target_values(ytr, choice["target_mode"]))
                prediction = price_predictions(model, Ava, choice["target_mode"])
                records.append({"Model": name, "fold": fold + 1, **regression_metrics(yva, prediction)})
    return pd.DataFrame(records, columns=columns)


def run_member1_pipeline(data_path=data_prep.DATA_PATH, output_dir="report/member1_checked", cv_folds=5):
    raw_df = data_prep.load_raw(data_path)
    cleaned_df = data_prep.clean(raw_df)
    train_df, validation_df, test_df = data_prep.split_dataframe(cleaned_df, seed=data_prep.SEED)
    processor = data_prep.TabularPreprocessor()
    X_train, y_train = processor.fit_transform(train_df)
    X_validation, y_validation = processor.transform(validation_df)
    X_test, y_test = processor.transform(test_df)
    # This scaler remains for the teammates' original feature matrix.
    X_train_scaled, X_validation_scaled, X_test_scaled, training_mean, training_scale = data_prep.standardize(
        X_train, X_validation, X_test)
    y_tier_train, threshold = data_prep.make_tier_label(y_train)
    y_tier_validation, _ = data_prep.make_tier_label(y_validation, threshold)
    y_tier_test, _ = data_prep.make_tier_label(y_test, threshold)
    print(f"\nRows raw/clean/train/validation/test: {len(raw_df)}/{len(cleaned_df)}/{len(train_df)}/{len(validation_df)}/{len(test_df)}")
    print(f"Encoded features: {X_train.shape[1]}; training tier threshold: {threshold:.2f} AZN")
    print("Validation search: raw/log-size inputs, price/log-price target, Ridge alpha")
    cached, selections, trials = {}, {}, []
    for feature_mode in ("raw", "log_size"):
        matrices = [linear_feature_matrix(X, processor.feature_names_, feature_mode)
                    for X in (X_train, X_validation, X_test)]
        cached[feature_mode] = data_prep.standardize(matrices[0], *matrices[1:])
        Atr, Ava = cached[feature_mode][:2]
        for target_mode in ("price", "log_price"):
            for family in ("Linear", "Ridge"):
                for alpha in ([0.0] if family == "Linear" else [0.01, 0.1, 1.0, 10.0, 100.0]):
                    model = _custom_model(family, alpha)
                    model.fit(Atr, target_values(y_train, target_mode))
                    try:
                        prediction = price_predictions(model, Ava, target_mode)
                        val_rmse = evaluate.rmse(y_validation, prediction)
                        train_rmse = evaluate.rmse(y_train, price_predictions(model, Atr, target_mode))
                    except FloatingPointError:
                        val_rmse, train_rmse = float("inf"), float("inf")
                    trial = {"family": family, "alpha": alpha, "feature_mode": feature_mode,
                             "target_mode": target_mode, "validation_RMSE": val_rmse, "training_RMSE": train_rmse}
                    trials.append(trial)
                    if np.isfinite(val_rmse) and (family not in selections or val_rmse < selections[family]["validation_RMSE"]):
                        selections[family] = trial.copy()
    if len(selections) != 2:
        raise RuntimeError("No finite validation configuration for a model family")
    # Freeze all choices before calculating any test metric.
    results, models_fitted, predictions, timed = {}, {}, {}, []
    for family, choice in selections.items():
        Atr, Ava, Ate = cached[choice["feature_mode"]][:3]
        models = {f"Custom {family}": _custom_model(family, choice["alpha"]),
                  f"sklearn {family}": LinearRegression() if family == "Linear" else Ridge(alpha=choice["alpha"], solver="svd")}
        for name, model in models.items():
            start = perf_counter()
            model.fit(Atr, target_values(y_train, choice["target_mode"]))
            fit_time = perf_counter() - start
            start = perf_counter()
            prediction = price_predictions(model, Ate, choice["target_mode"])
            predict_time = perf_counter() - start
            results[name] = regression_metrics(y_test, prediction)
            models_fitted[name], predictions[name] = model, prediction
            timed.append({"Model": name, **results[name], **choice, "fit_seconds": fit_time, "predict_seconds": predict_time})
        print(f"Selected {family}: {choice}")
    results["Training median"] = regression_metrics(y_test, np.full(len(y_test), np.median(y_train)))
    timed.append({"Model": "Training median", **results["Training median"]})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(timed).to_csv(output / "test_metrics.csv", index=False)
    pd.DataFrame(trials).to_csv(output / "validation_trials.csv", index=False)
    cv = _diagnostic_cv(train_df, selections, cv_folds, data_prep.SEED)
    cv.to_csv(output / "cv_folds.csv", index=False)
    if not cv.empty:
        summary = cv.groupby("Model")[["RMSE", "MAE", "R2"]].agg(["mean", "std"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary.to_csv(output / "cv_summary.csv")
    for family in ("Linear", "Ridge"):
        difference = np.max(np.abs(predictions[f"Custom {family}"] - predictions[f"sklearn {family}"]))
        print(f"Max Custom/sklearn {family} prediction difference: {difference:.6g} AZN")
    metadata = {"seed": data_prep.SEED, "cleaning_audit": cleaned_df.attrs["cleaning_audit"],
                "feature_names": processor.feature_names_, "selected_configs": selections,
                "tier_threshold_AZN": threshold, "split_method": "seeded random; not explicit tier stratification",
                "split_row_ids": {name: frame.source_row_id.astype(int).tolist() for name, frame in
                                  zip(("train", "validation", "test"), (train_df, validation_df, test_df))},
                "cv_note": "Training-only fixed-config descriptive CV, not nested CV.",
                "completed_scope": "Linear/Ridge only. Required tree/SVM integration is still separate."}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nTask A Linear/Ridge holdout results (AZN)")
    evaluate.compare(results)
    print(f"Tables saved to {output}")
    # Keep all keys that the uploaded run_all.py and teammates use.
    return {"preprocessor": processor, "feature_names": processor.feature_names_,
            "X_train": X_train, "X_validation": X_validation, "X_test": X_test,
            "X_train_scaled": X_train_scaled, "X_validation_scaled": X_validation_scaled, "X_test_scaled": X_test_scaled,
            "y_train": y_train, "y_validation": y_validation, "y_test": y_test,
            "y_tier_train": y_tier_train, "y_tier_validation": y_tier_validation, "y_tier_test": y_tier_test,
            "tier_threshold": threshold, "standardization_mean": training_mean, "standardization_scale": training_scale,
            "best_ridge_alpha": selections["Ridge"]["alpha"], "regression_results": results,
            "selections": selections, "linear_feature_cache": cached, "fitted_models": models_fitted}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(data_prep.DATA_PATH))
    parser.add_argument("--output", default="report/member1_checked")
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    run_member1_pipeline(args.data, args.output, args.cv_folds)

"""Training, explanation, and fairness audit pipeline."""

import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from fairlearn.metrics import MetricFrame, count, false_positive_rate, selection_rate
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.compose import ColumnTransformer
from sklearn import __version__ as sklearn_version
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             classification_report, log_loss, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import joblib


def run_pipeline(data: pd.DataFrame, target: str, sensitive: str, output_dir: str | Path,
                 test_size: float = 0.25, random_state: int = 42, threshold: float = 0.5,
                 positive_label=None, min_group_size: int = 30, bootstrap_samples: int = 1000,
                 time_column: str | None = None, target_fraction: float = 0.10,
                 contact_cost: float = 5.0, retained_value: float = 100.0,
                 intervention_success_rate: float = 0.20) -> dict:
    if data.empty:
        raise ValueError("The input dataset has no rows.")
    if data.columns.duplicated().any():
        raise ValueError("The input dataset contains duplicate column names; rename them before training.")
    if target not in data.columns:
        raise ValueError(f"Target column {target!r} was not found.")
    if data[target].isna().any():
        raise ValueError(f"Target column {target!r} contains missing values.")
    if sensitive not in data.columns:
        raise ValueError(f"Sensitive feature {sensitive!r} was not found in the dataset.")
    if sensitive == target:
        raise ValueError("The target and sensitive feature must be different columns.")
    if time_column and time_column not in data.columns:
        raise ValueError(f"Time column {time_column!r} was not found in the dataset.")
    if time_column in {target, sensitive}:
        raise ValueError("Time, target, and sensitive columns must be distinct.")
    if not 0 < target_fraction <= 1:
        raise ValueError("target_fraction must be greater than 0 and no more than 1.")
    if contact_cost < 0 or retained_value < 0 or not 0 <= intervention_success_rate <= 1:
        raise ValueError("contact_cost and retained_value must be nonnegative; intervention_success_rate must be in [0, 1].")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Sensitive attributes are deliberately excluded from model inputs.
    labels = list(pd.unique(data[target]))
    if len(labels) != 2:
        raise ValueError(f"Target {target!r} must contain exactly two classes; found {len(labels)}.")
    if positive_label is None:
        matches = [label for label in labels if str(label).strip().lower() in {"1", "true"}]
        if len(matches) != 1:
            raise ValueError(
                f"Cannot safely infer the positive target label from {labels!r}. "
                "Pass --positive-label with the class that means churn."
            )
        positive_label = matches[0]
    positive_matches = [label for label in labels if str(label) == str(positive_label)]
    if len(positive_matches) != 1:
        raise ValueError(f"Positive label {positive_label!r} must exactly match one target class: {labels!r}.")
    positive_label = positive_matches[0]
    y = (data[target].astype(str) == str(positive_label)).astype(int)
    class_counts = y.value_counts()
    if int(class_counts.min()) < 2:
        raise ValueError("Each target class needs at least two rows for a stratified train/test split.")
    X = data.drop(columns=[target, sensitive] + ([time_column] if time_column else []))
    all_missing = X.columns[X.isna().all()].tolist()
    if all_missing:
        X = X.drop(columns=all_missing)
    if X.shape[1] == 0:
        raise ValueError("No usable predictor columns remain after excluding target, sensitive, and all-missing columns.")
    sensitive_values = data[sensitive].astype("string").str.strip().replace("", pd.NA).fillna("Missing").astype(str)
    if time_column:
        timestamps = pd.to_datetime(data[time_column], errors="coerce", utc=True)
        if timestamps.isna().any():
            raise ValueError(f"Time column {time_column!r} contains values that cannot be parsed as dates.")
        latest = timestamps.max()
        is_test = timestamps.eq(latest)
        if is_test.all() or (~is_test).sum() == 0:
            raise ValueError("Temporal evaluation requires at least two distinct time periods.")
        X_train, X_test = X.loc[~is_test], X.loc[is_test]
        y_train, y_test = y.loc[~is_test], y.loc[is_test]
        sensitive_train, sensitive_test = sensitive_values.loc[~is_test], sensitive_values.loc[is_test]
    else:
        try:
            X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
                X, y, sensitive_values, test_size=test_size, random_state=random_state, stratify=y
            )
        except ValueError as exc:
            raise ValueError(f"Unable to create a stratified split: {exc}. Increase dataset size or test_size.") from exc
    if y_train.nunique() < 2:
        raise ValueError("The training split contains only one target class.")
    if y_test.nunique() < 2:
        raise ValueError("The test split contains only one target class; increase test_size or add examples.")

    try:
        X_fit, X_policy, y_fit, y_policy, sensitive_fit, sensitive_policy = train_test_split(
            X_train, y_train, sensitive_train, test_size=0.25,
            random_state=random_state, stratify=y_train,
        )
    except ValueError as exc:
        raise ValueError(f"Unable to reserve a mitigation validation split: {exc}. Add more training rows.") from exc
    numeric = X_fit.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in X_fit.columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    if not transformers:
        raise ValueError("No predictor columns remain after excluding target and sensitive feature.")
    def make_pipeline():
        preprocess = ColumnTransformer(transformers, verbose_feature_names_out=False)
        model = XGBClassifier(
            n_estimators=350, max_depth=4, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=2,
            reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=-1, random_state=random_state,
        )
        return Pipeline([("preprocess", preprocess), ("model", model)])

    def make_logistic_pipeline():
        logistic_transformers = []
        if numeric:
            logistic_transformers.append(("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric))
        if categorical:
            logistic_transformers.append(("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical))
        preprocess = ColumnTransformer(logistic_transformers, verbose_feature_names_out=False)
        return Pipeline([("preprocess", preprocess), ("model", LogisticRegression(
            max_iter=1500, solver="lbfgs", random_state=random_state,
        ))])

    # Keep separate fitting, postprocessor-validation, and held-out test sets.
    # The mitigation thresholds are selected on policy-validation data only.
    pipeline = make_pipeline().fit(X_fit, y_fit)
    logistic = make_logistic_pipeline().fit(X_fit, y_fit)
    minimum_training_class = int(y_fit.value_counts().min())
    if minimum_training_class < 2:
        raise ValueError("Calibration needs at least two training examples in each target class.")
    calibration_folds = min(5, minimum_training_class)
    calibrated = CalibratedClassifierCV(
        estimator=make_pipeline(), method="sigmoid", cv=calibration_folds
    ).fit(X_fit, y_fit)
    baseline = DummyClassifier(strategy="prior").fit(X_fit, y_fit)
    mitigation = ThresholdOptimizer(
        estimator=calibrated,
        constraints="demographic_parity",
        objective="accuracy_score",
        grid_size=1000,
        prefit=True,
        predict_method="predict_proba",
    )
    mitigation.fit(X_policy, y_policy, sensitive_features=sensitive_policy)
    joblib.dump(pipeline, output / "churn_pipeline.joblib")
    joblib.dump(logistic, output / "logistic_baseline.joblib")
    joblib.dump(calibrated, output / "calibrated_churn_pipeline.joblib")
    joblib.dump(mitigation, output / "fairlearn_threshold_optimizer.joblib")

    raw_probability = pipeline.predict_proba(X_test)[:, 1]
    probability = calibrated.predict_proba(X_test)[:, 1]
    baseline_probability = baseline.predict_proba(X_test)[:, 1]
    logistic_probability = logistic.predict_proba(X_test)[:, 1]
    prediction = (probability >= threshold).astype(int)
    raw_prediction = (raw_probability >= threshold).astype(int)
    mitigated_prediction = mitigation.predict(
        X_test, sensitive_features=sensitive_test, random_state=random_state
    )

    def score(probabilities):
        return {
            "accuracy_at_threshold": float(accuracy_score(y_test, probabilities >= threshold)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "average_precision": float(average_precision_score(y_test, probabilities)),
            "brier_score": float(brier_score_loss(y_test, probabilities)),
            "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
        }

    metrics = {
        "threshold": threshold,
        "test_rows": int(len(y_test)),
        "positive_label": str(positive_label),
        "models": {
            "prior_baseline": score(baseline_probability),
            "logistic_regression_baseline": score(logistic_probability),
            "xgboost_raw": score(raw_probability),
            "xgboost_sigmoid_calibrated": score(probability),
            "fairlearn_demographic_parity": {
                "accuracy": float(accuracy_score(y_test, mitigated_prediction)),
                "uses_group_specific_randomized_thresholds": True,
                "constraint": "demographic_parity",
            },
        },
        "calibrated_classification_report": classification_report(
            y_test, prediction, output_dict=True, zero_division=0
        ),
        "raw_xgboost_classification_report": classification_report(
            y_test, raw_prediction, output_dict=True, zero_division=0
        ),
    }
    economics = {
        "assumptions": {
            "target_fraction": target_fraction,
            "outreach_cost_per_customer": contact_cost,
            "retained_customer_value": retained_value,
            "intervention_success_rate": intervention_success_rate,
            "assumption_note": "Illustrative scenario inputs; replace with measured intervention costs, retention value, and causal lift before business use.",
        },
        "models": {
            "logistic_regression_baseline": _decision_economics(
                y_test.to_numpy(), logistic_probability, logistic_probability,
                target_fraction, contact_cost, retained_value, intervention_success_rate,
            ),
            "xgboost_raw_ranking_and_calibrated_probability": _decision_economics(
                y_test.to_numpy(), raw_probability, probability,
                target_fraction, contact_cost, retained_value, intervention_success_rate,
            ),
        },
        "interpretation": "Expected net value applies assumed intervention success to predicted churn risk; retrospective value applies the same assumed success to observed held-out churn labels. Neither estimates causal intervention lift.",
    }
    metrics["decision_economics"] = economics
    with (output / "model_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    pd.DataFrame({"actual_churn": y_test.to_numpy(), "raw_churn_probability": raw_probability,
                  "calibrated_churn_probability": probability, "baseline_probability": baseline_probability,
                  "logistic_churn_probability": logistic_probability,
                  "predicted_churn": prediction, "raw_predicted_churn": raw_prediction,
                  "fairlearn_mitigated_prediction": mitigated_prediction,
                  "sensitive_group": sensitive_test.to_numpy()},
                 index=X_test.index).to_csv(output / "predictions.csv", index_label="row_id")

    raw_observed, raw_mean = calibration_curve(y_test, raw_probability, n_bins=10, strategy="quantile")
    calibrated_observed, calibrated_mean = calibration_curve(y_test, probability, n_bins=10, strategy="quantile")
    baseline_observed, baseline_mean = calibration_curve(y_test, baseline_probability, n_bins=10, strategy="quantile")
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(raw_mean, raw_observed, marker="o", label="XGBoost raw")
    plt.plot(calibrated_mean, calibrated_observed, marker="o", label="XGBoost sigmoid calibrated")
    plt.plot(baseline_mean, baseline_observed, marker="o", label="Prior baseline")
    plt.xlabel("Mean predicted churn probability")
    plt.ylabel("Observed churn rate")
    plt.title("Calibration on held-out test set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "calibration_curve.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Fairlearn reports group-specific rates plus max-minus-min disparities.
    metrics_by_group = _group_metrics(y_test, prediction, sensitive_test)
    mitigated_group_metrics = _group_metrics(y_test, mitigated_prediction, sensitive_test)
    standard_audit = _fairness_summary(
        metrics_by_group, y_test.to_numpy(), prediction, sensitive_test.to_numpy(),
        bootstrap_samples, random_state, min_group_size,
    )
    mitigated_audit = _fairness_summary(
        mitigated_group_metrics, y_test.to_numpy(), mitigated_prediction, sensitive_test.to_numpy(),
        bootstrap_samples, random_state, min_group_size,
    )
    observed_gender = sensitive_test.to_numpy() != "Missing"
    observed_gender_audits = {}
    if observed_gender.any() and len(pd.unique(sensitive_test.to_numpy()[observed_gender])) >= 2:
        known_y = y_test.to_numpy()[observed_gender]
        known_groups = sensitive_test.to_numpy()[observed_gender]
        observed_gender_audits = {
            "standard_threshold_known_gender_only": _fairness_summary(
                _group_metrics(known_y, prediction[observed_gender], known_groups),
                known_y, prediction[observed_gender], known_groups,
                bootstrap_samples, random_state + 1, min_group_size,
            ),
            "demographic_parity_known_gender_only": _fairness_summary(
                _group_metrics(known_y, mitigated_prediction[observed_gender], known_groups),
                known_y, mitigated_prediction[observed_gender], known_groups,
                bootstrap_samples, random_state + 1, min_group_size,
            ),
            "excluded_missing_rows": int((~observed_gender).sum()),
            "note": "Conditional comparison among records with observed gender; does not correct bias from non-random missingness or establish fairness for customers with missing gender.",
        }
    fairness = {
        "sensitive_feature": sensitive,
        "standard_threshold": standard_audit,
        "fairlearn_demographic_parity_mitigation": mitigated_audit,
        "known_gender_only_sensitivity": observed_gender_audits,
        "mitigation": "ThresholdOptimizer selects group-specific randomized thresholds using a separate policy-validation split.",
        "interpretation": "A parity constraint is a chosen trade-off, not a universal definition of fairness or evidence of legal compliance.",
    }
    with (output / "fairness_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(fairness, handle, indent=2)
    pd.concat({"standard_threshold": metrics_by_group.by_group,
               "fairlearn_demographic_parity_mitigation": mitigated_group_metrics.by_group},
              names=["approach", "group"]).to_csv(output / "fairness_by_group.csv")
    metadata = {
        "target_column": target, "positive_label": str(positive_label), "sensitive_column": sensitive,
        "model_features": list(X.columns), "excluded_sensitive_feature": sensitive,
        "dropped_all_missing_features": all_missing,
        "input_rows": int(len(data)), "test_size": test_size, "random_state": random_state,
        "split_strategy": "latest_time_period_holdout" if time_column else "stratified_random_holdout",
        "time_column": time_column,
        "test_period": str(pd.to_datetime(data.loc[X_test.index, time_column], utc=True).max()) if time_column else None,
        "fit_rows": int(len(X_fit)), "mitigation_validation_rows": int(len(X_policy)),
        "classification_threshold": threshold, "calibration": "sigmoid, cross-validated on training split only",
        "calibration_folds": calibration_folds, "bootstrap_samples": bootstrap_samples,
        "fairness_mitigation": "Fairlearn ThresholdOptimizer demographic_parity; randomized group-specific thresholds",
        "decision_economics": economics["assumptions"],
        "minimum_group_size_warning": min_group_size, "python_version": platform.python_version(),
        "scikit_learn_version": sklearn_version,
    }
    with (output / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    # SHAP explains the fitted tree model on preprocessed test rows; values are raw margins (log-odds).
    transformed = pipeline.named_steps["preprocess"].transform(X_test)
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    explain_frame = pd.DataFrame(transformed, columns=names, index=X_test.index)
    explain_frame = explain_frame.iloc[: min(len(explain_frame), 1000)]
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    explanation = explainer(explain_frame)
    shap_values = np.asarray(explanation.values)
    mean_abs = np.abs(shap_values).mean(axis=0)
    pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    ).to_csv(output / "shap_feature_importance.csv", index=False)
    plt.figure()
    shap.summary_plot(shap_values, explain_frame, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(output / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()
    return {"metrics": metrics, "fairness": fairness, "output_dir": str(output)}


def _true_positive_rate(y_true, y_pred):
    from sklearn.metrics import recall_score
    return recall_score(y_true, y_pred, zero_division=0)


def _group_metrics(y_true, prediction, groups):
    return MetricFrame(
        metrics={"count": count, "selection_rate": selection_rate,
                 "true_positive_rate": _true_positive_rate,
                 "false_positive_rate": false_positive_rate,
                 "accuracy": accuracy_score},
        y_true=y_true, y_pred=prediction, sensitive_features=groups,
    )


def _fairness_summary(frame, y_true, prediction, groups, bootstrap_samples, random_state, min_group_size):
    y_array = np.asarray(y_true, dtype=int)
    group_array = np.asarray(groups).astype(str)
    group_label_prevalence = {}
    for group in sorted(pd.unique(group_array), key=str):
        mask = group_array == group
        group_label_prevalence[str(group)] = {
            "count": int(mask.sum()),
            "positive_count": int(y_array[mask].sum()),
            "positive_rate": float(y_array[mask].mean()) if mask.any() else None,
        }
    return {
        "group_metrics": _json_safe(frame.by_group.astype(float).to_dict(orient="index")),
        "group_label_prevalence": group_label_prevalence,
        "difference_max_minus_min": _json_safe(frame.difference().astype(float).to_dict()),
        "ratio_min_over_max": _json_safe(frame.ratio().astype(float).to_dict()),
        "confidence_intervals_95_percent": _fairness_uncertainty(
            y_true, prediction, groups, bootstrap_samples, random_state
        ),
        "low_sample_groups": [str(group) for group, group_count in pd.Series(groups).value_counts().items()
                              if group_count < min_group_size],
        "minimum_group_size_for_warning": min_group_size,
    }


def _json_safe(value):
    """Replace undefined fairness ratios (for example 0/0) with JSON null."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _fairness_uncertainty(y_true, y_pred, groups, samples, seed):
    """Percentile bootstrap intervals, resampling rows within each observed group.

    This is conditional on the fitted model and held-out predictions. It does not
    capture training, policy-selection, repeated-customer, or missingness uncertainty.
    """
    if samples < 50:
        raise ValueError("At least 50 bootstrap samples are required for uncertainty intervals.")
    rng = np.random.default_rng(seed)
    group_values = np.asarray(sorted(pd.unique(groups), key=str))
    metric_names = ("selection_rate", "true_positive_rate", "false_positive_rate", "accuracy")
    group_draws = {str(group): {metric: [] for metric in metric_names} for group in group_values}
    difference_draws = {metric: [] for metric in metric_names}

    def rates(actual, predicted):
        positives = actual == 1
        negatives = actual == 0
        return {
            "selection_rate": float(np.mean(predicted)) if len(predicted) else np.nan,
            "true_positive_rate": float(np.mean(predicted[positives])) if positives.any() else np.nan,
            "false_positive_rate": float(np.mean(predicted[negatives])) if negatives.any() else np.nan,
            "accuracy": float(np.mean(actual == predicted)) if len(actual) else np.nan,
        }

    group_indices = {str(group): np.flatnonzero(groups == group) for group in group_values}
    for _ in range(samples):
        replicate = {}
        for group, indices in group_indices.items():
            draw = rng.choice(indices, size=len(indices), replace=True)
            values = rates(y_true[draw], y_pred[draw])
            replicate[group] = values
            for metric, value in values.items():
                if np.isfinite(value):
                    group_draws[group][metric].append(value)
        for metric in metric_names:
            values = [row[metric] for row in replicate.values() if np.isfinite(row[metric])]
            if values:
                difference_draws[metric].append(max(values) - min(values))

    def interval(values):
        if not values:
            return [None, None]
        return [float(v) for v in np.percentile(values, [2.5, 97.5])]

    return {
        "method": "Percentile bootstrap resampling test rows within each observed group; intervals are descriptive, not simultaneous.",
        "conditional_on_fixed_model_and_predictions": True,
        "does_not_capture": ["model-training uncertainty", "mitigation-policy selection uncertainty", "repeated-customer dependence", "non-random missingness"],
        "bootstrap_samples": samples,
        "group_rates": {group: {metric: interval(values) for metric, values in metrics.items()}
                        for group, metrics in group_draws.items()},
        "difference_max_minus_min": {metric: interval(values) for metric, values in difference_draws.items()},
    }


def _decision_economics(y_true, ranking_scores, expected_churn_probability, target_fraction,
                        contact_cost, retained_value, intervention_success_rate):
    """Illustrative capacity-constrained outreach scenario; not causal ROI."""
    y_true = np.asarray(y_true, dtype=int)
    ranking_scores = np.asarray(ranking_scores, dtype=float)
    expected_churn_probability = np.asarray(expected_churn_probability, dtype=float)
    if not (len(y_true) == len(ranking_scores) == len(expected_churn_probability)):
        raise ValueError("Economic scoring arrays must have the same length.")
    if len(y_true) == 0 or not np.isfinite(ranking_scores).all() or not np.isfinite(expected_churn_probability).all():
        raise ValueError("Economic scoring requires nonempty finite scores and probabilities.")
    if ((expected_churn_probability < 0) | (expected_churn_probability > 1)).any():
        raise ValueError("Expected churn probabilities must be in [0, 1].")

    selected_count = max(1, int(np.ceil(len(y_true) * target_fraction)))
    selected = np.argsort(-ranking_scores, kind="stable")[:selected_count]
    selected_labels = y_true[selected]
    selected_probabilities = expected_churn_probability[selected]
    total_churn = int(y_true.sum())
    expected_saves = float(selected_probabilities.sum() * intervention_success_rate)
    random_expected_saves = float(expected_churn_probability.sum() * (selected_count / len(y_true)) * intervention_success_rate)
    cost = float(selected_count * contact_cost)
    expected_net_value = expected_saves * retained_value - cost
    random_net_value = random_expected_saves * retained_value - cost

    return {
        "selection_rule": "Select the highest-risk customers by ranking score up to the fixed capacity; ties resolved by stable input order.",
        "selected_count": selected_count,
        "selected_fraction": float(selected_count / len(y_true)),
        "observed_churners_selected": int(selected_labels.sum()),
        "observed_churn_capture_rate": float(selected_labels.sum() / total_churn) if total_churn else None,
        "observed_churn_rate_in_selected": float(selected_labels.mean()),
        "expected_churners_in_selected": float(selected_probabilities.sum()),
        "expected_successful_retains_under_assumed_lift": expected_saves,
        "assumed_outreach_cost": cost,
        "illustrative_expected_net_value": expected_net_value,
        "random_targeting_illustrative_expected_net_value_same_capacity": random_net_value,
        "incremental_expected_net_value_vs_random_targeting": float(expected_net_value - random_net_value),
        "observed_label_scenario_net_value_not_causal": float(selected_labels.sum() * intervention_success_rate * retained_value - cost),
        "warning": "Scenario arithmetic only. Intervention success rate and retained value are assumptions; predictions are not causal treatment effects and may be miscalibrated.",
    }

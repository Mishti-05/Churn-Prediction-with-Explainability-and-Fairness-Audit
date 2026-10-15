# Model card: churn risk demo

## Summary

This project demonstrates binary churn prediction with XGBoost, global SHAP explanations, Fairlearn group assessment, and an optional Fairlearn `ThresholdOptimizer` postprocessing comparison.

## Intended use

- Learning, prototyping, and demonstrating model evaluation and fairness trade-offs.
- Exploring how feature-based predictions and group-level outcome metrics change under one explicitly selected parity constraint.

## Out of scope

- Making automatic customer, pricing, eligibility, service, or retention decisions.
- Claiming that a model is fair, unbiased, causally valid, legally compliant, or production-ready.
- Treating the generated synthetic dataset as evidence about real customers.

## Data and labels

The default data generator creates fictional telecom-like records. The project also supports a supplied KKBox competition archive. Its preparation workflow builds sampled February and March customer cohorts from `train.csv`, `train_v2.csv`, transaction history, and member attributes. The model feature table excludes raw member IDs; gender is excluded from the base model's features and retained for group assessment and mitigation. Listening logs are not used in this first temporal extension. For other CSVs, the caller specifies a binary target, sensitive group, and optionally a time column. The target's positive class must be explicit unless it is `1` or `true`.

The project does not currently validate real-world data provenance, consent, representativeness, label quality, repeated customer identity, temporal leakage, or the appropriateness of group definitions. Correlated proxy features may encode sensitive information.

## Model and evaluation

The main learner is an XGBoost binary classifier with median/mode imputation and one-hot encoding. A standardized logistic regression with the same feature split and imputation/encoding is included as a simple predictive baseline; a prior-only dummy baseline is included as well. A sigmoid probability calibrator is cross-validated on the fit split. For KKBox, the February 2017 expiration cohort is used as training and the March 2017 cohort is held out as an out-of-time test set. Within the February cohort, separate fit and policy-validation subsets support calibration and postprocessing. The policy subset is used by Fairlearn to select postprocessing thresholds. The held-out March cohort is used for reported model and group metrics.

The demonstration compares prior-only and logistic-regression baselines, raw and sigmoid-calibrated XGBoost at a single global threshold, and Fairlearn demographic-parity postprocessing. Metrics include ranking quality, probability quality, and classification performance. A separate fixed-capacity scenario ranks the top 10% of the held-out cohort and calculates illustrative net value using stated outreach cost, retained-customer value, and intervention-success assumptions. The assumptions are configurable. This retrospective calculation is not a causal intervention-effect estimate and is especially sensitive to probability miscalibration.

For the recorded 150,000-row March holdout, logistic regression reached ROC AUC 0.820 and average precision 0.460; raw XGBoost reached 0.884 and 0.570. The sigmoid-calibrated model reached 0.883 ROC AUC and 0.927 accuracy. Under default, arbitrary value-unit assumptions, top-10% raw-XGBoost ranking produced a modeled net value of 44,602 units versus -55,470 units for random targeting at the same capacity. That estimate is illustrative only and must not be represented as measured retention savings.

The all-category selection-rate gap was 2.72 percentage points (95% CI 2.44–2.99) at the standard threshold and 0.027 points (0.028–0.317) after demographic-parity postprocessing. The TPR gap changed from 2.70 points (1.34–4.55) to 16.87 points (15.42–18.38). These intervals are conditional, descriptive bootstrap intervals; the model card limitations below qualify their scope.

## Fairness intervention and trade-offs

Fairlearn's `ThresholdOptimizer` postprocesses the calibrated predictor with group-specific randomized thresholds to target demographic parity while optimizing accuracy on the policy-validation data. Demographic parity is one possible constraint, not a universal definition of fairness. Matching selection rates can change error rates or accuracy across groups. Results depend on sample sizes, labels, threshold policy, and group definitions. The report gives percentile-bootstrap 95% intervals for group rates and max-minus-min gaps, plus a sensitivity comparison limited to rows with observed gender.

This mitigation requires the sensitive group at prediction time and can randomize decisions. That may be inappropriate or impermissible in some contexts. Review the deployment context, applicable law and policy, and affected-community impact before any use. The intervention is included for comparison and education, not as a recommendation to deploy.

## Explanations

SHAP summary values explain the fitted tree model's raw output (log-odds) on up to 1,000 held-out examples. They describe model behavior, not causal effects or a reason a particular customer will churn. The SHAP view explains the base XGBoost model, not the Fairlearn postprocessor's group-specific decision rule.

## Known limitations

- Synthetic data is fictional and deliberately simple.
- The KKBox result uses 150,000 sampled records per cohort and only transaction and member features; listening logs are omitted.
- Gender is missing for 89,857 of the 150,000 March test rows (about 60%). Missingness may be non-random and the available gender field may be incomplete or unreliable. The all-record audit therefore treats "Missing" as its own observed category; the known-gender-only sensitivity audit excludes those customers and cannot establish fairness for them or fix missing-not-at-random bias. No conclusion should be generalized to a complete or verified gender attribute.
- The out-of-time evaluation spans one month. Repeated customers can occur across cohorts, representing follow-up prediction for existing subscribers; customer identity itself is excluded from features and exports.
- Default bootstrap intervals use 1,000 within-group resamples. They are descriptive, not simultaneous guarantees, and condition on fixed test predictions; they omit model-training, postprocessor-selection, repeat-customer dependence, and non-random-missingness uncertainty. Very small groups or few positive/negative outcomes can make subgroup rates and intervals unstable.
- The project does not test subgroup intersections, drift, calibration by group, privacy risk, or intervention effects.
- The baseline and models are not tuned or independently externally validated.
- Sensitive attributes are written to prediction/audit outputs; apply appropriate access controls to real data.

## Human oversight

Treat outputs as exploratory evidence. A qualified multidisciplinary review should assess data quality, model validity, group definitions, metric choice, uncertainty, disparate impact, privacy, and the consequences of false positives and false negatives before considering any operational use.

## Reproducibility

Run configuration and software version metadata are written to `artifacts/run_metadata.json`. The supplied run held out the 2017-02-28 feature cohort and reports its observed metrics in the README. Metadata records the economic scenario assumptions, Python and scikit-learn versions, and bootstrap sample count but is not a complete environment lockfile. Preserve the input data version and installed dependency versions when reproducing a result.

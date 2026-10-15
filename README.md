# Churn prediction, explanations, and fairness audit

A runnable reference project that trains an **XGBoost** binary classifier, explains its behavior with **SHAP**, and measures group-level outcomes with **Fairlearn**, Microsoft's open-source fairness toolkit.

Run immediately on reproducible synthetic telecom data or provide a CSV. The sensitive attribute is excluded from the model inputs and reserved for auditing; correlated proxy features may still carry group information.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .
python -m churn_audit.cli
```

The run writes these files to `artifacts/`:

- `model_metrics.json`: compares a prior baseline, standardized logistic-regression baseline, raw XGBoost, and calibrated XGBoost on accuracy, ROC AUC, average precision, Brier score, and log loss. It also reports a capacity-constrained top-risk targeting scenario.
- `predictions.csv`: actual label, raw and calibrated churn probabilities, thresholded predictions, baseline probability, and audited group.
- `churn_pipeline.joblib`: fitted preprocessing and XGBoost pipeline.
- `calibrated_churn_pipeline.joblib`: cross-validated sigmoid-calibrated pipeline.
- `fairlearn_threshold_optimizer.joblib`: demographic-parity postprocessor fitted on a separate policy-validation split.
- `run_metadata.json`: label mapping, feature list, split/threshold settings, and Python/scikit-learn versions.
- `calibration_curve.png`: raw and calibrated XGBoost plus prior baseline calibration curves.
- `shap_feature_importance.csv` and `shap_summary.png`: global SHAP impacts. Values explain the model's raw margin (log-odds).
- `fairness_metrics.json` and `fairness_by_group.csv`: compare the calibrated global threshold with Fairlearn demographic-parity postprocessing, with group counts, rates, differences, ratios, and 95% within-group bootstrap intervals for gaps. A separate known-gender-only sensitivity audit highlights the large missing-gender segment.

## Portfolio materials

- [Model card](MODEL_CARD.md).
- [Five-minute demo guide](DEMO.md) and [six-slide presentation](presentations/Churn_Prediction_Explainability_Fairness_KKBox_Interview_Final.pptx).
- [Aggregate KKBox sample report](reports/kkbox-sampled-evaluation/README.md). It excludes row-level predictions, source data, and serialized models.

## Run on your CSV

```bash
python -m churn_audit.cli --data data/customers.csv --target churn --sensitive region --output artifacts
```

The CLI also exposes the illustrative fixed-capacity outreach scenario. Defaults target the top 10% by each model's risk ranking, with assumed outreach cost `$5`, retained-customer value `$100`, and success rate `20%`. Replace these with measured business values; the output is scenario arithmetic, not a causal estimate of intervention lift:

```powershell
python -m churn_audit.cli --data data/customers.csv --target churn --sensitive region `
  --target-fraction 0.10 --contact-cost 5 --retained-value 100 `
  --intervention-success-rate 0.20 --bootstrap-samples 1000
```

The target must have exactly two classes. If classes are `0` and `1` (or `false` and `true`), `1`/`true` is used as churn. For any other labels, explicitly identify the churn class with `--positive-label`; the other class becomes 0. This avoids silently treating an arbitrary alphabetically sorted class as churn. The prediction threshold defaults to 0.5 and can be changed to inspect threshold effects:

```bash
python -m churn_audit.cli --data data/customers.csv --target churn --positive-label Yes --sensitive age_band --threshold 0.4
```

## KKBox out-of-time evaluation

The supplied KKBox competition bundle can be prepared without extracting the multi-gigabyte listening logs. The preparation step extracts only label, transaction, and member files, creates point-in-time February and March customer snapshots, and omits raw member IDs from the feature table. By default it samples 150,000 labels per cohort with stratification for a manageable run; use `--max-rows-per-cohort 0` to process the full labeled cohorts.

```powershell
python -m churn_audit.prepare_kkbox --archive "$env:USERPROFILE\OneDrive\Desktop\projects\kkbox-churn-prediction-challenge.zip"
python -m churn_audit.cli --data data/kkbox/kkbox_temporal_features.csv --target is_churn --positive-label 1 --sensitive gender --time-column feature_cutoff --output artifacts/kkbox-temporal
```

The February cohort uses transaction history through 2017-01-31; the March cohort uses history through 2017-02-28. The latest snapshot is held out as the out-of-time test set. Features summarize historical transaction activity and member attributes; listening logs are not used. Raw extracted files and feature data stay under the ignored `data/kkbox/` directory and are not added to version control. Check the competition's data-use terms before sharing source data or outputs.

### Dataset sharing notice

This repository does not include the KKBox competition archive, extracted customer data, row-level predictions, or serialized model binaries. The [official competition rules](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/rules) permit public sharing of code under an eligible open-source license, while restricting access to the competition data itself. Obtain the data through Kaggle under your own account, accept and follow its rules, and do not upload or redistribute the dataset. This project is presented as an educational evaluation; aggregate results are specific to the sampled data and should not be treated as a substitute for reviewing the rules.

## License

Project code and documentation are released under the [MIT License](LICENSE). This does not grant rights to the KKBox competition data or third-party dependencies.

## KKBox demo run (150,000 rows per cohort)

The updated run on the supplied bundle used 112,500 February-cohort rows for fitting, 37,500 for calibration and policy validation, and 150,000 March-cohort rows for final evaluation. Logistic regression scored ROC AUC **0.820**, average precision **0.460**, and accuracy **0.919**. Raw XGBoost scored ROC AUC **0.884**, average precision **0.570**, and accuracy **0.924**. The sigmoid-calibrated XGBoost scored ROC AUC **0.883** and accuracy **0.927**. Calibration slightly reduced ROC AUC and did not improve Brier score (**0.0578** calibrated vs **0.0563** raw) or log loss (**0.2121** vs **0.2010**) on this holdout.

The default top-10% targeting scenario selects 15,000 of 150,000 test customers. Ranking by raw XGBoost and valuing churn probability with its sigmoid-calibrated probabilities gives **5,980 expected churners** among those selected. Under illustrative assumptions of 20% intervention success, 100 value units per retained customer and 5 units outreach cost, modeled net value is **44,602 units**, versus **−55,470** for random targeting at the same capacity (incremental scenario value **100,072 units**). On observed labels, this top decile captured **59.1%** of the holdout churners. Logistic regression captured **50.5%** and its scenario incremental value over random targeting was **90,697 units**. These are hypothetical scenario calculations, not estimated causal lift or actual savings; calibration was imperfect, and the assumptions must be replaced with measured values.

Fairlearn's demographic-parity postprocessor reduced the all-record selection-rate gap by gender from **2.72 pp** (95% bootstrap CI **2.44–2.99 pp**) to **0.027 pp** (CI **0.028–0.317 pp**). Accuracy changed from **0.927** to **0.923**. The true-positive-rate gap widened from **2.70 pp** (CI **1.34–4.55 pp**) to **16.87 pp** (CI **15.42–18.38 pp**). These percentile-bootstrap intervals use 1,000 within-group resamples and condition on the fitted models, selected policy, and held-out predictions; they do not cover training/policy uncertainty, repeat-customer dependence, or missing-not-at-random effects. The “Missing” gender group contains **89,857 (59.9%)** of test rows and is reported separately. Observed churn prevalence was **6.25%** in that category and about **13.1%** in each recorded gender group. The recorded female and male groups are not tiny by row count (28,454 and 31,689), but the high missingness makes the fairness audit incomplete. These findings are specific to this sampled cohort and chosen features, split, threshold, and parity objective; they are not a fairness certification or deployment recommendation.

## Interpreting results

Calibration uses sigmoid/Platt scaling with cross-validation on training data only; final comparison is on the held-out test split. Brier score and log loss measure probability quality (lower is better). A calibration curve compares probability bins with observed churn frequency. The simple baseline predicts the training-set churn prevalence for every customer.


The mitigation comparison uses Fairlearn `ThresholdOptimizer` with a demographic-parity constraint and accuracy objective. Thresholds are selected on a separate validation portion of training data and evaluated on the held-out test set. It uses group-specific randomized thresholds, so it needs the sensitive group at prediction time. The selected parity constraint is a deliberate trade-off, not a universal fairness definition. The ordinary global-threshold model remains reported as a comparison. See [MODEL_CARD.md](MODEL_CARD.md) for limitations and [DEMO.md](DEMO.md) for a short presentation walkthrough.

Fairlearn's `MetricFrame` reports both approaches on the same held-out test set. Difference is max minus min across groups; ratio is min divided by max. Selection rate is the share predicted to churn. True positive rate is recall among customers who churned. Percentile-bootstrap 95% intervals resample test rows within each group, and groups below `--min-group-size` (default 30) are flagged. The default is 1,000 resamples. These intervals condition on fixed model predictions and do not capture training, policy-selection, repeated-customer, or non-random missingness uncertainty; they are descriptive, not simultaneous guarantees. When a sensitive attribute has substantial missingness, the separate observed-value-only comparison is a sensitivity analysis, not a correction for missing-not-at-random data.

The synthetic dataset and its intentionally modest group effect are for demonstration only. SHAP describes model behavior, not causal effects. Production use needs domain validation, calibration, privacy controls, monitoring, and human review.


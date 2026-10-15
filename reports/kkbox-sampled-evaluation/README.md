# Aggregate sample evaluation

Reproduced from the locally supplied KKBox feature table on the February-to-March 2017 temporal split. The 150,000-row test cohort is the March period. See [run metadata](run_metadata.json), [model metrics and illustrative economics](model_metrics.json), and [Fairlearn audit with 1,000-resample intervals](fairness_metrics.json).

Only aggregate metrics, group summaries, global feature importance, and plots are included. Customer-level predictions, source data, and serialized fitted models are excluded. Do not add the competition dataset here; obtain it under the [Kaggle competition rules](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/rules).

The top-10% value calculation uses example assumptions, is sensitive to probability calibration, and is not a causal estimate of saved customers or revenue. Fairness intervals are conditional on fixed model/policy predictions and do not account for non-random missing gender or repeat-customer dependence.

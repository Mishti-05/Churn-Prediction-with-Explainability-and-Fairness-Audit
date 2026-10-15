# 5-minute demo guide

## Prepare

Install the project and prepare the supplied KKBox bundle:

```bash
python -m pip install -e .
python -m churn_audit.prepare_kkbox --archive "$env:USERPROFILE\OneDrive\Desktop\projects\kkbox-churn-prediction-challenge.zip"
python -m churn_audit.cli --data data/kkbox/kkbox_temporal_features.csv --target is_churn --positive-label 1 --sensitive gender --time-column feature_cutoff --output artifacts/kkbox-temporal-extended --bootstrap-samples 1000 --target-fraction 0.10 --contact-cost 5 --retained-value 100 --intervention-success-rate 0.20
```

## Walkthrough

1. **Set the scope:** This is anonymized competition data with a one-month historical holdout, not evidence about current customers.
2. **Inspect model performance:** Open `artifacts/kkbox-temporal-extended/model_metrics.json`. Compare prior-only, logistic-regression, raw XGBoost, and calibrated XGBoost results. Brier score and log loss assess probability quality; lower is better. In the recorded run, logistic ROC AUC was 0.820 and raw XGBoost ROC AUC was 0.884; calibration did not improve Brier score or log loss over raw XGBoost.
3. **Explain the predictor:** Open `artifacts/kkbox-temporal-extended/shap_summary.png` and note that it explains base-model behavior in log-odds, not causes.
4. **Compare group outcomes:** Open `artifacts/kkbox-temporal-extended/fairness_by_group.csv` and `artifacts/kkbox-temporal-extended/fairness_metrics.json`. Compare the ordinary global threshold with Fairlearn demographic-parity postprocessing, including group counts and bootstrap confidence intervals.
5. **Review the capacity-based business scenario:** In `model_metrics.json`, inspect the top-10% capture rate and illustrative net value against random targeting. Challenge the assumed success rate, retention value and outreach cost. The scenario calculations are not a causal estimate of intervention lift.
6. **Discuss the trade-off:** Demographic parity targets similar positive prediction rates. Compare what happened to accuracy, true positive rate, and false positive rate. Do not call one metric a complete fairness verdict.
7. **State the deployment caveat:** The postprocessor needs the sensitive group at prediction time and may randomize predictions. This is a demo intervention, not an automatic deployment recommendation.

In the recorded run, demographic parity narrowed the selection-rate gap from 2.72 pp (95% CI 2.44–2.99) to 0.03 pp (CI 0.03–0.32), but widened the TPR gap from 2.70 pp (CI 1.34–4.55) to 16.87 pp (CI 15.42–18.38) and reduced accuracy from 0.927 to 0.923. Which trade-offs matter depends on the use case and affected people.

## Questions to invite

- Which outcome is meaningful for customers and the business?
- Which fairness criteria and group definitions are justified in this context?
- Would a future-time split or customer-level split change the result?
- What are the impacts of false positives and false negatives?
- What additional evidence and governance would be needed before any real use?

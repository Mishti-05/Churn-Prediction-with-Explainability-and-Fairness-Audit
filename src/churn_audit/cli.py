"""Command line entry point."""

import argparse
import json

from .data import load_data
from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train a churn model, explain it with SHAP, and audit group fairness with Fairlearn.")
    parser.add_argument("--data", help="CSV path. Omit to use the synthetic telecom demo.")
    parser.add_argument("--target", default="churn", help="Binary target column (default: churn).")
    parser.add_argument("--positive-label", help="Target value that means churn; auto-detected only for 1/true labels.")
    parser.add_argument("--sensitive", default="region", help="Group column to audit (default: region).")
    parser.add_argument("--time-column", help="Optional date column; holds out its latest time period for evaluation.")
    parser.add_argument("--output", default="artifacts", help="Directory for generated results.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for positive churn predictions.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Held-out test fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--min-group-size", type=int, default=30, help="Warn when a test-set group has fewer rows.")
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Bootstrap replicates for descriptive 95%% fairness intervals (minimum 50).")
    parser.add_argument("--target-fraction", type=float, default=0.10, help="Share of customers to target in the capacity-constrained economics scenario (default: 0.10).")
    parser.add_argument("--contact-cost", type=float, default=5.0, help="Illustrative outreach cost per targeted customer.")
    parser.add_argument("--retained-value", type=float, default=100.0, help="Illustrative value of one retained customer.")
    parser.add_argument("--intervention-success-rate", type=float, default=0.20, help="Assumed intervention success/lift in [0,1]; scenario only, not estimated causally.")
    args = parser.parse_args()
    if not 0 < args.test_size < 1 or not 0 < args.threshold < 1:
        parser.error("--test-size and --threshold must be between 0 and 1.")
    if args.min_group_size < 1 or args.bootstrap_samples < 50:
        parser.error("--min-group-size must be positive and --bootstrap-samples must be at least 50.")
    if not 0 < args.target_fraction <= 1:
        parser.error("--target-fraction must be in (0,1].")
    if args.contact_cost < 0 or args.retained_value < 0 or not 0 <= args.intervention_success_rate <= 1:
        parser.error("--contact-cost and --retained-value must be nonnegative; --intervention-success-rate must be in [0,1].")
    data = load_data(args.data, args.target, args.seed)
    result = run_pipeline(data, args.target, args.sensitive, args.output, args.test_size, args.seed,
                          args.threshold, args.positive_label, args.min_group_size, args.bootstrap_samples,
                          args.time_column, args.target_fraction, args.contact_cost,
                          args.retained_value, args.intervention_success_rate)
    print(json.dumps({"metrics": {k: v for k, v in result["metrics"].items() if "classification_report" not in k},
                      "fairness_difference": {
                          name: audit["difference_max_minus_min"]
                          for name, audit in result["fairness"].items()
                          if isinstance(audit, dict) and "difference_max_minus_min" in audit
                      },
                      "artifacts": result["output_dir"]}, indent=2))


if __name__ == "__main__":
    main()

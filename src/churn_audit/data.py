"""Data loading and synthetic demo data generation."""

from pathlib import Path

import numpy as np
import pandas as pd


def make_demo_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Create a reproducible, fictional telecom churn dataset for a quick demo."""
    rng = np.random.default_rng(random_state)
    tenure = rng.integers(0, 73, n_samples)
    monthly_charges = np.clip(rng.normal(72, 22, n_samples), 18, 130)
    support_calls = np.clip(rng.poisson(1.5, n_samples), 0, 9)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n_samples, p=[0.56, 0.25, 0.19])
    internet = rng.choice(["Fiber optic", "DSL", "No"], n_samples, p=[0.48, 0.40, 0.12])
    payment = rng.choice(["Electronic check", "Bank transfer", "Credit card", "Mailed check"], n_samples)
    region = rng.choice(["North", "South", "East", "West"], n_samples, p=[0.32, 0.28, 0.24, 0.16])
    # Region carries a modest synthetic disparity to make the audit useful as a demo.
    region_effect = pd.Series(region).map({"North": 0.00, "South": 0.08, "East": -0.03, "West": 0.12}).to_numpy()
    logit = (
        1.15
        - 0.055 * tenure
        + 0.021 * (monthly_charges - 70)
        + 0.44 * support_calls
        + np.where(contract == "Month-to-month", 0.85, np.where(contract == "One year", -0.35, -0.9))
        + np.where(internet == "Fiber optic", 0.25, np.where(internet == "No", -0.2, 0.0))
        + np.where(payment == "Electronic check", 0.3, 0.0)
        + region_effect
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)
    return pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charges": monthly_charges.round(2),
        "support_calls": support_calls,
        "contract": contract,
        "internet_service": internet,
        "payment_method": payment,
        "region": region,
        "churn": churn,
    })


def load_data(path: str | Path | None, target: str, random_state: int = 42) -> pd.DataFrame:
    if path is None:
        data = make_demo_data(random_state=random_state)
    else:
        data = pd.read_csv(path)
    if target not in data.columns:
        raise ValueError(f"Target column {target!r} was not found. Columns: {list(data.columns)}")
    if data[target].isna().any():
        raise ValueError(f"Target column {target!r} contains missing values.")
    if data[target].nunique() != 2:
        raise ValueError(f"Target column {target!r} must contain exactly two classes.")
    return data

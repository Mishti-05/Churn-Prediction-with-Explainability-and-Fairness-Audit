"""Extract selected KKBox files and build leakage-aware February/March cohorts."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from pathlib import PurePosixPath

import numpy as np
import pandas as pd
import py7zr
from sklearn.model_selection import train_test_split


ARCHIVE_FILES = ("train.csv.7z", "train_v2.csv.7z", "transactions.csv.7z", "members_v3.csv.7z")
TRANSACTION_COLUMNS = (
    "msno", "transaction_date", "payment_plan_days", "plan_list_price",
    "actual_amount_paid", "is_auto_renew", "is_cancel",
)
MEMBER_COLUMNS = ("msno", "city", "bd", "gender", "registered_via", "registration_init_time")


def extract_sources(competition_zip: Path, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(competition_zip) as bundle:
        available = set(bundle.namelist())
        missing = [name for name in ARCHIVE_FILES if name not in available]
        if missing:
            raise ValueError(f"Competition archive is missing required files: {missing}")
        for archive_name in ARCHIVE_FILES:
            csv_name = archive_name.removesuffix(".7z")
            output_csv = raw_dir / csv_name
            if output_csv.exists() and output_csv.stat().st_size > 0:
                continue
            nested_7z = raw_dir / archive_name
            temporary = nested_7z.with_suffix(nested_7z.suffix + ".partial")
            with bundle.open(archive_name) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            temporary.replace(nested_7z)
            try:
                with py7zr.SevenZipFile(nested_7z, mode="r") as archive:
                    names = archive.getnames()
                    matching = [name for name in names if PurePosixPath(name).name == csv_name]
                    if len(matching) != 1:
                        raise ValueError(f"Expected {csv_name!r} inside {archive_name}; found {names[:8]}")
                    archive.extract(path=raw_dir, targets=matching)
            finally:
                nested_7z.unlink(missing_ok=True)
            extracted_csv = raw_dir.joinpath(*PurePosixPath(matching[0]).parts)
            if not extracted_csv.exists():
                raise RuntimeError(f"Extraction did not create expected file: {output_csv}")
            if extracted_csv != output_csv:
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                extracted_csv.replace(output_csv)
            print(f"Extracted {csv_name}: {output_csv.stat().st_size:,} bytes")


def _sample_labels(path: Path, max_rows: int | None, seed: int) -> pd.DataFrame:
    labels = pd.read_csv(path, dtype={"msno": "string", "is_churn": "int8"})
    if labels["msno"].duplicated().any():
        labels = labels.drop_duplicates("msno", keep="last")
    if max_rows and len(labels) > max_rows:
        labels, _ = train_test_split(
            labels, train_size=max_rows, random_state=seed, stratify=labels["is_churn"]
        )
    return labels.reset_index(drop=True)


def _aggregate_chunk(frame: pd.DataFrame, cutoff: pd.Timestamp, suffix: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    aggregation = {"transaction_date_int": ("transaction_date_int", "max")}
    for column in ("payment_plan_days", "plan_list_price", "actual_amount_paid", "is_auto_renew", "is_cancel"):
        aggregation[f"{column}_sum"] = (column, "sum")
        aggregation[f"{column}_count"] = (column, "count")
    grouped = frame.groupby("msno", sort=False).agg(
        transaction_count=("msno", "size"), **aggregation
    )
    grouped = grouped.rename(columns={
        "transaction_date_int": f"last_transaction_date_int{suffix}",
        **{col: f"{col}{suffix}" for col in grouped.columns if col != "transaction_date_int"},
    })
    return grouped


def _merge_stats(left: pd.DataFrame | None, right: pd.DataFrame) -> pd.DataFrame:
    if right.empty:
        return left if left is not None else right
    if left is None:
        return right
    sum_columns = [c for c in right.columns if c not in {"last_transaction_date_int", "last_transaction_date_int_90d"}]
    combined = left[sum_columns].add(right[sum_columns], fill_value=0)
    for column in set(left.columns) - set(sum_columns):
        combined[column] = pd.concat([left[column], right[column]], axis=1).max(axis=1)
    return combined


def build_transaction_features(path: Path, cohort_ids: set[str], cutoff_text: str) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff_text)
    full = recent = None
    usecols = list(TRANSACTION_COLUMNS)
    dtypes = {"msno": "string", "transaction_date": "string"}
    for chunk in pd.read_csv(path, usecols=usecols, dtype=dtypes, chunksize=500_000):
        chunk = chunk[chunk["msno"].isin(cohort_ids)].copy()
        if chunk.empty:
            continue
        chunk["transaction_date_int"] = pd.to_numeric(chunk["transaction_date"], errors="coerce")
        cutoff_int = int(cutoff.strftime("%Y%m%d"))
        eligible = chunk[chunk["transaction_date_int"].le(cutoff_int)].copy()
        if eligible.empty:
            continue
        numeric = ("payment_plan_days", "plan_list_price", "actual_amount_paid", "is_auto_renew", "is_cancel")
        for column in numeric:
            eligible[column] = pd.to_numeric(eligible[column], errors="coerce")
        full = _merge_stats(full, _aggregate_chunk(eligible, cutoff, ""))
        recent_rows = eligible[eligible["transaction_date_int"].ge(int((cutoff - pd.Timedelta(days=89)).strftime("%Y%m%d")))]
        recent = _merge_stats(recent, _aggregate_chunk(recent_rows, cutoff, "_90d"))
    if full is None:
        return pd.DataFrame(index=pd.Index([], name="msno"))

    features = full.copy()
    if recent is not None:
        features = features.join(recent, how="left")
    features["days_since_last_transaction"] = pd.to_datetime(
        features["last_transaction_date_int"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
    ).rsub(cutoff).dt.days
    if "last_transaction_date_int_90d" in features:
        features["transactions_90d"] = features.pop("transaction_count_90d")
        features["days_since_last_transaction_90d"] = pd.to_datetime(
            features.pop("last_transaction_date_int_90d").astype("Int64").astype(str),
            format="%Y%m%d", errors="coerce",
        ).rsub(cutoff).dt.days
    for suffix in ("", "_90d"):
        for column in ("payment_plan_days", "plan_list_price", "actual_amount_paid", "is_auto_renew", "is_cancel"):
            sums, counts = f"{column}_sum{suffix}", f"{column}_count{suffix}"
            if sums in features:
                features[f"{column}_mean{suffix}"] = features[sums] / features[counts].replace(0, np.nan)
                features.drop(columns=[sums, counts], inplace=True)
    features.drop(columns=[c for c in ("last_transaction_date_int",) if c in features], inplace=True)
    return features


def load_members(path: Path, all_ids: set[str]) -> tuple[pd.DataFrame, pd.Series]:
    matches = []
    for chunk in pd.read_csv(path, usecols=list(MEMBER_COLUMNS), dtype={"msno": "string", "gender": "string", "registration_init_time": "string"}, chunksize=500_000):
        selected = chunk[chunk["msno"].isin(all_ids)]
        if not selected.empty:
            matches.append(selected)
    if not matches:
        return pd.DataFrame(columns=MEMBER_COLUMNS).set_index("msno")
    members = pd.concat(matches, ignore_index=True).drop_duplicates("msno", keep="last").set_index("msno")
    members["bd"] = pd.to_numeric(members["bd"], errors="coerce")
    members.loc[~members["bd"].between(10, 100), "bd"] = np.nan
    registration = pd.to_datetime(members["registration_init_time"], format="%Y%m%d", errors="coerce")
    members.drop(columns=["registration_init_time"], inplace=True)
    # Express registration recency against each row's own prediction cutoff.
    return members, registration


def build_feature_table(raw_dir: Path, output_csv: Path, max_rows_per_cohort: int | None, seed: int) -> dict:
    v1 = _sample_labels(raw_dir / "train.csv", max_rows_per_cohort, seed)
    v2 = _sample_labels(raw_dir / "train_v2.csv", max_rows_per_cohort, seed + 1)
    cohorts = ((v1, "2017-01-31"), (v2, "2017-02-28"))
    all_ids = set(pd.concat([v1["msno"], v2["msno"]], ignore_index=True).astype(str))
    members, registration = load_members(raw_dir / "members_v3.csv", all_ids)
    tables = []
    for labels, cutoff in cohorts:
        features = build_transaction_features(raw_dir / "transactions.csv", set(labels["msno"].astype(str)), cutoff)
        frame = labels.set_index("msno").join(features, how="left").join(members, how="left")
        reg = registration.reindex(frame.index)
        frame["registration_age_days"] = (pd.Timestamp(cutoff) - reg).dt.days
        frame["feature_cutoff"] = cutoff
        tables.append(frame.reset_index(drop=True))
        print(f"Built {cutoff} cohort: {len(frame):,} rows; positive rate={frame['is_churn'].mean():.4f}")
    result = pd.concat(tables, ignore_index=True)
    # No customer identifier is exported; it is only an internal join key.
    result.drop(columns=["msno"], errors="ignore", inplace=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return {"output": str(output_csv), "rows": len(result), "cohorts": [cutoff for _, cutoff in cohorts],
            "max_rows_per_cohort": max_rows_per_cohort, "sensitive_feature": "gender",
            "time_column": "feature_cutoff", "feature_columns": [c for c in result.columns if c not in {"is_churn", "gender", "feature_cutoff"}]}


def main():
    parser = argparse.ArgumentParser(description="Prepare a point-in-time KKBox churn feature table from the supplied Kaggle archive.")
    parser.add_argument("--archive", required=True, type=Path, help="Path to kkbox-churn-prediction-challenge.zip")
    parser.add_argument("--data-dir", type=Path, default=Path("data/kkbox"))
    parser.add_argument("--output", type=Path, default=Path("data/kkbox/kkbox_temporal_features.csv"))
    parser.add_argument("--max-rows-per-cohort", type=int, default=150_000,
                        help="Stratified cap for a practical first run; use 0 for all rows.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw_dir = args.data_dir / "raw"
    extract_sources(args.archive, raw_dir)
    cap = args.max_rows_per_cohort or None
    receipt = build_feature_table(raw_dir, args.output, cap, args.seed)
    receipt_path = args.output.with_suffix(".metadata.json")
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd


@dataclass
class ImportResult:
    df: pd.DataFrame
    warnings: List[str]
    detected_columns: dict


def _find_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _to_numeric_series(s: pd.Series) -> pd.Series:
    # Converts currency/strings like "1,234.50" to float safely
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def detect_meta_export(df_raw: pd.DataFrame) -> bool:
    """
    Lightweight detection for Meta Ads Manager exports.
    """
    # Common date columns
    date_candidates = ["Reporting starts", "Day", "Date", "Reporting ends"]
    # Common spend columns (currency varies)
    spend_candidates = [c for c in df_raw.columns if "Amount spent" in c] + ["Spend", "Cost"]
    # Results often appears
    has_results = "Results" in df_raw.columns

    has_date = _find_first_existing_column(df_raw, date_candidates) is not None
    has_spend = _find_first_existing_column(df_raw, spend_candidates) is not None

    return bool(has_date and has_spend and has_results)

def import_meta_export(
    df_raw: pd.DataFrame,
    *,
    default_value_per_conversion: float | None = None,
) -> ImportResult:
    """
    Converts Meta export into canonical daily schema:
    date, spend, conversions, value_per_conversion, net_value
    """
    warnings: List[str] = []
    detected_columns: dict = {}

    # 1) Identify date column
    # Prefer true daily columns first
    date_col = _find_first_existing_column(df_raw, ["Day", "Date"])
    if not date_col:
        # Fallback to period columns (campaign-level exports)
        date_col = _find_first_existing_column(df_raw, ["Reporting starts", "Reporting ends"])

    if not date_col:
        raise ValueError(
            "Meta import failed: couldn't find a date column "
            "(Reporting starts/Day/Date/Reporting ends)."
        )
    detected_columns["date_col"] = date_col

    # 2) Identify spend column (Meta uses "Amount spent (INR)" etc.)
    spend_col = None
    for c in df_raw.columns:
        if "Amount spent" in c:
            spend_col = c
            break
    if not spend_col:
        spend_col = _find_first_existing_column(df_raw, ["Spend", "Cost"])
    if not spend_col:
        raise ValueError("Meta import failed: couldn't find spend column (Amount spent / Spend / Cost).")
    detected_columns["spend_col"] = spend_col

    # 3) Identify conversions column
    conv_col = _find_first_existing_column(df_raw, ["Results"])
    if not conv_col:
        raise ValueError("Meta import failed: couldn't find conversions column (Results).")
    detected_columns["conversions_col"] = conv_col

    # Build working df
    df = df_raw[[date_col, spend_col, conv_col]].copy()
    df.columns = ["date", "spend", "conversions"]

    # 4) Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = int(df["date"].isna().sum())
    if bad_dates > 0:
        warnings.append(f"{bad_dates} rows have invalid dates and were dropped.")
        df = df.dropna(subset=["date"])

    # 5) Parse numeric
    df["spend"] = _to_numeric_series(df["spend"]).fillna(0.0)
    df["conversions"] = _to_numeric_series(df["conversions"]).fillna(0.0)

    # 6) Aggregate daily (Meta exports are often campaign-level rows per day)
    df = df.groupby("date", as_index=False).agg({"spend": "sum", "conversions": "sum"})

    # 7) Value per conversion & net value
    if default_value_per_conversion is None:
        raise ValueError(
            "Meta export does not include conversion value. "
            "Please provide Default Value per Conversion in the app."
        )

    df["value_per_conversion"] = float(default_value_per_conversion)
    df["net_value"] = (df["conversions"] * df["value_per_conversion"]) - df["spend"]

    # Data quality warnings
    if len(df) < 5:
        warnings.append(
            f"Only {len(df)} day(s) of data detected after aggregation. "
            "Decision confidence may be unstable; ideally use 14–30 days."
        )

    if float(df["spend"].sum()) == 0.0:
        warnings.append("Total spend is 0 after parsing. Check if spend column is correct.")

    if float(df["conversions"].sum()) == 0.0:
        warnings.append("Total conversions is 0 after parsing. Check if 'Results' is correct.")

    return ImportResult(df=df, warnings=warnings, detected_columns=detected_columns)
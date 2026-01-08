from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

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
    # Handles "1,234.50", currency symbols, etc.
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def detect_google_export(df_raw: pd.DataFrame) -> bool:
    """
    Lightweight detection for Google Ads exports.
    We detect by presence of a date column + cost + conversions-like column.
    """
    date_candidates = ["Date", "Day"]
    cost_candidates = ["Cost", "Cost (INR)", "Cost (USD)", "Cost micros", "Amount"]
    conv_candidates = ["Conversions", "All conversions"]

    has_date = _find_first_existing_column(df_raw, date_candidates) is not None
    has_cost = _find_first_existing_column(df_raw, cost_candidates) is not None
    has_conversions = _find_first_existing_column(df_raw, conv_candidates) is not None

    return bool(has_date and has_cost and has_conversions)


def import_google_export(
    df_raw: pd.DataFrame,
    *,
    default_value_per_conversion: float | None = None,
) -> ImportResult:
    """
    Converts Google Ads export into canonical daily schema:
    date, spend, conversions, value_per_conversion, net_value
    """
    warnings: List[str] = []
    detected: dict = {}

    # 1) Date
    date_col = _find_first_existing_column(df_raw, ["Date", "Day"])
    if not date_col:
        raise ValueError("Google import failed: couldn't find a date column (Date/Day).")
    detected["date_col"] = date_col

    # 2) Spend (Cost)
    cost_col = _find_first_existing_column(
        df_raw, ["Cost", "Cost (INR)", "Cost (USD)", "Amount", "Cost micros"]
    )
    if not cost_col:
        raise ValueError("Google import failed: couldn't find spend column (Cost / Cost micros).")
    detected["spend_col"] = cost_col

    # 3) Conversions
    conv_col = _find_first_existing_column(df_raw, ["Conversions", "All conversions"])
    if not conv_col:
        raise ValueError("Google import failed: couldn't find conversions column (Conversions/All conversions).")
    detected["conversions_col"] = conv_col

    # 4) Optional conversion value
    value_col = _find_first_existing_column(
        df_raw,
        ["Conversion value", "Conv. value", "Conv. value (by conv. time)", "Conversion value (by conv. time)"],
    )
    if value_col:
        detected["conversion_value_col"] = value_col
    else:
        detected["conversion_value_col"] = None
        if default_value_per_conversion is None:
            warnings.append(
                "Google export has no conversion value column. Using default value per conversion from the app."
            )

    # Build working df
    cols = [date_col, cost_col, conv_col] + ([value_col] if value_col else [])
    df = df_raw[cols].copy()

    # Normalize column names
    rename_map = {date_col: "date", cost_col: "spend", conv_col: "conversions"}
    if value_col:
        rename_map[value_col] = "conversion_value"
    df = df.rename(columns=rename_map)

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = int(df["date"].isna().sum())
    if bad_dates > 0:
        warnings.append(f"{bad_dates} rows have invalid dates and were dropped.")
        df = df.dropna(subset=["date"])

    # Parse numerics
    df["spend"] = _to_numeric_series(df["spend"]).fillna(0.0)
    df["conversions"] = _to_numeric_series(df["conversions"]).fillna(0.0)

    if "conversion_value" in df.columns:
        df["conversion_value"] = _to_numeric_series(df["conversion_value"]).fillna(0.0)

    # Handle Cost micros (if present)
    if "micros" in str(cost_col).lower():
        # Convert micros to currency
        df["spend"] = df["spend"] / 1_000_000.0
        warnings.append("Detected Cost micros; converted micros to currency units.")

    # Aggregate daily (Google exports can contain multiple rows per day)
    agg_map = {"spend": "sum", "conversions": "sum"}
    if "conversion_value" in df.columns:
        agg_map["conversion_value"] = "sum"

    df = df.groupby("date", as_index=False).agg(agg_map)

    # Compute value_per_conversion + net_value
    if "conversion_value" in df.columns and df["conversions"].sum() > 0:
        # Use actual value column to derive average value per conversion
        # (conversion_value is total value; value_per_conversion is average)
        df["value_per_conversion"] = 0.0
        mask = df["conversions"] > 0
        df.loc[mask, "value_per_conversion"] = df.loc[mask, "conversion_value"] / df.loc[mask, "conversions"]
    else:
        if default_value_per_conversion is None:
            raise ValueError(
                "Google export does not include conversion value and no Default Value per Conversion was provided."
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
        warnings.append("Total spend is 0 after parsing. Check if Cost column is correct.")

    if float(df["conversions"].sum()) == 0.0:
        warnings.append("Total conversions is 0 after parsing. Check if Conversions column is correct.")

    # If conversion_value existed, drop it from canonical schema
    if "conversion_value" in df.columns:
        df = df.drop(columns=["conversion_value"])

    return ImportResult(df=df, warnings=warnings, detected_columns=detected)

# mdu_engine/validation.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


@dataclass
class ValidationResult:
    is_valid: bool
    status: str  # DECISION_OK | DECISION_BLOCKED
    block_reason: Optional[str]
    warnings: List[str]
    metrics: Dict[str, Any]


def validate_normalized_daily_schema(df_norm: pd.DataFrame) -> ValidationResult:
    """
    Validates the normalized daily schema expected by the engine:
      date, spend, conversions, value_per_conversion, net_value

    Returns a ValidationResult that can be used to:
      - block decisions (industry standard)
      - show warnings
      - log audit metrics
    """
    warnings: List[str] = []
    metrics: Dict[str, Any] = {}

    required_cols = ["date", "spend", "conversions", "net_value"]
    missing = [c for c in required_cols if c not in df_norm.columns]
    if missing:
        return ValidationResult(
            is_valid=False,
            status="DECISION_BLOCKED",
            block_reason=f"Normalized data missing required columns: {missing}",
            warnings=[],
            metrics={"missing_columns": missing},
        )

    # Parse dates
    date_series = pd.to_datetime(df_norm["date"], errors="coerce")
    valid_dates = date_series.dropna()

    if valid_dates.empty:
        return ValidationResult(
            is_valid=False,
            status="DECISION_BLOCKED",
            block_reason="No valid dates found after parsing. Please export a daily report (Breakdown: Day).",
            warnings=[],
            metrics={"days_of_data": 0, "date_invalid_rate": 1.0},
        )

    days_of_data = int(valid_dates.dt.date.nunique())
    date_min = valid_dates.min().date().isoformat()
    date_max = valid_dates.max().date().isoformat()

    metrics.update(
        {
            "days_of_data": days_of_data,
            "date_min": date_min,
            "date_max": date_max,
            "rows": int(len(df_norm)),
            "unique_dates": days_of_data,
        }
    )

    # Invalid date rate (helps identify broken exports)
    date_invalid_rate = float(date_series.isna().mean())
    metrics["date_invalid_rate"] = date_invalid_rate
    if date_invalid_rate > 0.05:
        warnings.append(f"High invalid date rate: {date_invalid_rate:.1%}. Export may be malformed.")

    # Spend checks
    spend = pd.to_numeric(df_norm["spend"], errors="coerce")
    spend_invalid_rate = float(spend.isna().mean())
    metrics["spend_invalid_rate"] = spend_invalid_rate

    spend_sum = float(spend.fillna(0).sum())
    metrics["spend_total"] = spend_sum

    if spend_invalid_rate > 0.05:
        warnings.append(f"High invalid spend rate: {spend_invalid_rate:.1%}.")
    if spend_sum <= 0:
        return ValidationResult(
            is_valid=False,
            status="DECISION_BLOCKED",
            block_reason="Spend total is 0 (or missing). Decisioning requires non-zero spend in the selected window.",
            warnings=warnings,
            metrics=metrics,
        )

    if (spend.fillna(0) < 0).any():
        return ValidationResult(
            is_valid=False,
            status="DECISION_BLOCKED",
            block_reason="Negative spend detected. Please export a standard performance report (daily).",
            warnings=warnings,
            metrics=metrics,
        )

    # Conversions checks
    conv = pd.to_numeric(df_norm["conversions"], errors="coerce")
    conv_invalid_rate = float(conv.isna().mean())
    metrics["conversions_invalid_rate"] = conv_invalid_rate

    conv_sum = float(conv.fillna(0).sum())
    metrics["conversions_total"] = conv_sum

    if conv_invalid_rate > 0.10:
        warnings.append(f"High invalid conversions rate: {conv_invalid_rate:.1%}.")
    if (conv.fillna(0) < 0).any():
        return ValidationResult(
            is_valid=False,
            status="DECISION_BLOCKED",
            block_reason="Negative conversions detected. Please export a clean daily performance report.",
            warnings=warnings,
            metrics=metrics,
        )

    # Daily-ness check (industry expectation)
    # If only 1 unique day, warn strongly and block if it's clearly aggregated
    if days_of_data < 7:
        return ValidationResult(
            is_valid=False,
            status="DECISION_BLOCKED",
            block_reason=f"Insufficient daily data window: {days_of_data} day(s). Provide at least 7 daily rows (7–30 recommended).",
            warnings=warnings,
            metrics=metrics,
        )

    # Duplicate-date density check (warn, not block)
    dup_dates = int(valid_dates.dt.date.duplicated().sum())
    metrics["duplicate_date_rows"] = dup_dates
    if dup_dates > 0:
        warnings.append(
            f"Duplicate dates detected ({dup_dates} row(s)). Consider exporting one row per day or ensure the importer aggregates cleanly."
        )

    return ValidationResult(
        is_valid=True,
        status="DECISION_OK",
        block_reason=None,
        warnings=warnings,
        metrics=metrics,
    )


def validation_to_dict(v: ValidationResult) -> Dict[str, Any]:
    return asdict(v)
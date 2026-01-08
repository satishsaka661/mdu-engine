from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from mdu_engine.importers.meta import (
    detect_meta_export,
    import_meta_export,
    ImportResult,
)

from mdu_engine.importers.google import (
    detect_google_export,
    import_google_export,
)

@dataclass
class RoutedImport:
    platform: str
    result: ImportResult


def route_import(
    df_raw: pd.DataFrame,
    *,
    default_value_per_conversion: float | None = None
) -> RoutedImport:

    if detect_meta_export(df_raw):
        return RoutedImport(
            platform="meta",
            result=import_meta_export(
                df_raw,
                default_value_per_conversion=default_value_per_conversion,
            ),
        )

    if detect_google_export(df_raw):
        return RoutedImport(
            platform="google",
            result=import_google_export(
                df_raw,
                default_value_per_conversion=default_value_per_conversion,
            ),
        )

    raise ValueError(
    "Could not detect platform format.\n\n"
    "Supported:\n"
    "- Meta Ads Manager exports (Reporting starts/Day, Amount spent, Results)\n"
    "- Google Ads performance exports (Date/Day, Cost, Conversions)\n\n"
    "Tip: Your Google file looks like a campaign settings export. "
    "Download a performance report segmented by Day with Cost + Conversions."
)

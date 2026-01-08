import numpy as np
import pandas as pd


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df["net_value"] = (df["conversions"] * df["value_per_conversion"]) - df["spend"]
    return df


def compute_decision_confidence(
    df: pd.DataFrame,
    signal_reliability: float = 0.6,
    scale_pct: float = 0.10,
    simulations: int = 5000,
    random_seed: int | None = None,
) -> dict:
    # Ensure net_value exists (for Streamlit uploads)
    if "net_value" not in df.columns:
        df = df.copy()
        df["net_value"] = (df["conversions"] * df["value_per_conversion"]) - df["spend"]

    # Historical stats
    avg_nv = float(df["net_value"].mean())
    std_nv = float(df["net_value"].std(ddof=1))
    n_days = int(len(df))

    # Guardrails for tiny datasets
    if n_days < 3 or std_nv == 0:
        return {
            "avg_net_value": round(avg_nv, 2),
            "decision_confidence": 0.0,
            "downside_risk": 1.0,
            "note": "Not enough variance/data to estimate uncertainty. Add more days.",
            "random_seed": random_seed,
        }

    # Standard error (uncertainty of mean)
    se = std_nv / np.sqrt(n_days)

    # ✅ Deterministic RNG (industry standard)
    rng = np.random.default_rng(random_seed)

    # Simulate future net value (uncertainty-aware)
    simulated_nv = rng.normal(loc=avg_nv, scale=se, size=simulations)

    # Simple scaling effect:
    # assume only ~60% of the spend change translates into net value improvement
    scale_effect = avg_nv * scale_pct * 0.6
    simulated_nv_scaled = simulated_nv + scale_effect

    # Success condition: scaled outcome not worse than baseline average
    success = simulated_nv_scaled >= avg_nv

    raw_confidence = float(success.mean())
    decision_confidence = raw_confidence * float(signal_reliability)

    downside_risk = float((simulated_nv_scaled < avg_nv).mean())

    return {
        "avg_net_value": round(avg_nv, 2),
        "decision_confidence": round(decision_confidence, 3),
        "downside_risk": round(downside_risk, 3),
        "simulations": simulations,
        "signal_reliability": signal_reliability,
        "scale_pct": scale_pct,
        "random_seed": random_seed,
    }

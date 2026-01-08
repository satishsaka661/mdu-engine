import streamlit as st
import io
import pandas as pd
import hashlib

from mdu_engine.decision_confidence import compute_decision_confidence
from mdu_engine.decision_rules import RISK_PROFILES, decide_action
from mdu_engine.reporting import recommendation_summary, write_markdown_report
from mdu_engine.importers.router import route_import
from mdu_engine.history import log_decision, read_history
from mdu_engine.version import ENGINE_VERSION, RULESET_VERSION

from mdu_engine.validation import validate_normalized_daily_schema, validation_to_dict

# NEW (portfolio)
from mdu_engine.portfolio_decision import ChannelDecision, recommend_portfolio_action


# -----------------------------
# Robust CSV reader
# -----------------------------
def read_ads_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()

    header_idx = None
    sep = ","

    for i, line in enumerate(lines[:200]):
        l = line.strip()

        if not l or l.lower().startswith("total:") or "Total:" in l:
            continue

        # Google signals
        if (l.startswith("Day,") or l.startswith("Date,")) and ("Cost" in l) and ("Conversions" in l):
            header_idx = i
            sep = ","
            break

        # Meta signals
        if ("Amount spent" in l) and ("Results" in l) and (("Day" in l) or ("Reporting starts" in l) or ("Date" in l)):
            header_idx = i
            sep = ","
            break

    if header_idx is None:
        for i, line in enumerate(lines[:200]):
            l = line.strip()
            if any(ch.isalpha() for ch in l) and ("," in l or "\t" in l) and "Total:" not in l:
                header_idx = i
                sep = "\t" if l.count("\t") > l.count(",") else ","
                break

    if header_idx is None:
        raise ValueError("Could not detect a valid header row in this CSV.")

    return pd.read_csv(
        io.BytesIO(raw),
        skiprows=header_idx,
        sep=sep,
        engine="python",
    )
def stable_seed_from_df(df: pd.DataFrame) -> int:
    """
    Creates a deterministic seed from the normalized dataframe content.
    Same data -> same seed -> reproducible Monte Carlo.
    """
    df2 = df.copy()
    # keep only key columns in a stable order
    cols = [c for c in ["date", "spend", "conversions", "value_per_conversion", "net_value"] if c in df2.columns]
    df2 = df2[cols].copy()

    # normalize types/format
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce").dt.date.astype(str)
    for c in cols:
        if c != "date":
            df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0.0).round(6)

    payload = df2.to_csv(index=False).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()

    # fit into 32-bit range for numpy seed compatibility
    return int(digest[:8], 16)

# -----------------------------
# Run pipeline for ONE file
# -----------------------------
def process_uploaded_file(
    uploaded_file,
    default_vpc: float,
    signal_reliability: float,
    scale_pct: float,
    simulations: int,
    profile
):
    df_raw = read_ads_csv(uploaded_file)

    routed = route_import(
        df_raw,
        default_value_per_conversion=default_vpc if default_vpc > 0 else None
    )
    import_result = routed.result
    df_norm = import_result.df

    # ✅ VALIDATION (industry standard) - BEFORE Monte Carlo
    v = validate_normalized_daily_schema(df_norm)
    v_dict = validation_to_dict(v)

    # Always build a result dict with window metrics (for UI/report)
    result = {
        "date_min": v.metrics.get("date_min"),
        "date_max": v.metrics.get("date_max"),
        "days_of_data": v.metrics.get("days_of_data", 0),
        "spend_total": float(v.metrics.get("spend_total", df_norm["spend"].sum() if "spend" in df_norm else 0.0)),
        "validation": v_dict,
        "engine_version": ENGINE_VERSION,
        "ruleset_version": RULESET_VERSION,
        # safe placeholders to avoid crashes in downstream code
        "decision_confidence": 0.0,
        "downside_risk": 1.0,
        "avg_net_value": float(df_norm["net_value"].mean()) if "net_value" in df_norm and len(df_norm) else 0.0,
    }

    # If blocked: do NOT run compute_decision_confidence
    if not v.is_valid:
        decision = {
            "status": "DECISION_BLOCKED",
            "confidence_tier": "n/a",
            "action": "HOLD",
            "recommended_change_pct_range": "0% (no change)",
            "next_review_window": "After fixing export",
            "reason": v.block_reason,
            "user_explanation": (
                "I can’t make a reliable scale/reduce decision because the export/data window is not suitable. "
                "Fix the export (daily breakdown, 7–30 days) and re-upload."
            ),
            "explainability": {
                "what_happened": [v.block_reason],
                "what_could_go_wrong": [
                    "Acting on insufficient or malformed data can lead to incorrect budget changes."
                ],
                "what_to_do_next": [
                    "Export a daily report (Breakdown: Day) with at least 7 days (7–30 recommended) and re-upload."
                ],
            }
        }
        return routed.platform, df_raw, import_result, result, decision

    # ✅ Monte Carlo only if validation passed
    seed = stable_seed_from_df(df_norm)
    result["random_seed"] = seed
    mc = compute_decision_confidence(
        df_norm,
        signal_reliability=signal_reliability,
        scale_pct=scale_pct,
        simulations=simulations,
    )
    result.update(mc)
    result["simulations"] = simulations
    result["signal_reliability"] = signal_reliability
    result["scale_pct"] = scale_pct

    decision = decide_action(
        decision_confidence=result["decision_confidence"],
        downside_risk=result["downside_risk"],
        profile=profile,
        days_of_data=result.get("days_of_data"),
    )

    return routed.platform, df_raw, import_result, result, decision


# -----------------------------
# Build ChannelDecision for portfolio
# -----------------------------
def build_channel_decision(label: str, import_result, result: dict, decision: dict) -> ChannelDecision:
    df_norm = import_result.df
    spend_total = float(result.get("spend_total", df_norm["spend"].sum()))
    avg_net_value = float(result.get("avg_net_value", df_norm["net_value"].mean()))

    # Portfolio module typically expects SCALE/HOLD/REDUCE.
    # If MAINTAIN exists, map it safely to HOLD.
    action = decision.get("action", "HOLD")
    if action == "MAINTAIN":
        action = "HOLD"

    return ChannelDecision(
        platform=label,
        decision=action,
        avg_net_value=avg_net_value,
        downside_risk=float(result.get("downside_risk", 1.0)),
        confidence=float(result.get("decision_confidence", 0.0)),
        spend_total=spend_total,
        notes=tuple(import_result.warnings or []),
    )


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("MDU Engine")
with st.expander("About / Health", expanded=False):
    st.write("MDU Engine — Decision Engine for Meta Ads + Google Ads")
    st.write(f"Engine Version: {ENGINE_VERSION}")
    st.write(f"Ruleset Version: {RULESET_VERSION}")
    st.write("Status: Running ✅")

with st.expander("✅ Upload Instructions (V1 formats only)", expanded=True):
    st.markdown("""
### Meta Ads (Daily Performance Export)
**Required**
- Breakdown by **Day**
- Columns:
  - `Date` / `Day` / `Reporting starts`
  - `Amount spent`
  - `Results` (Leads / Purchases)

---

### Google Ads (Campaign Performance Report)
**Required**
- Report segmented by **Day**
- Columns:
  - `Date` / `Day`
  - `Cost`
  - `Conversions` **or** `All conversions`

---

### General Rules
- One row per day (or data that can aggregate cleanly to daily)
- Mixed platforms in one CSV ❌
- Totals rows will be ignored automatically
""")

col1, col2 = st.columns(2)
with col1:
    uploaded_meta = st.file_uploader("Upload Meta CSV", type=["csv"], key="meta")
with col2:
    uploaded_google = st.file_uploader("Upload Google CSV", type=["csv"], key="google")

profile_name = st.selectbox("Risk Profile", ["balanced", "conservative", "growth"])
signal_reliability = st.slider("Signal Reliability", 0.0, 1.0, 0.6, 0.05)
scale_pct = st.slider("Scale % (for simulation)", 0.01, 0.50, 0.10, 0.01)
simulations = st.selectbox("Simulations", [1000, 3000, 5000, 10000], index=2)

default_vpc = st.number_input(
    "Default Value per Conversion (used if CSV has no conversion value column)",
    min_value=0.0,
    value=5000.0,
    step=100.0
)

if not uploaded_meta and not uploaded_google:
    st.info("Upload at least one CSV (Meta and/or Google) to proceed.")
    st.stop()

profile = RISK_PROFILES[profile_name]

channel_outputs = {}
channels_for_portfolio = {}

# Process Meta
if uploaded_meta:
    try:
        out = process_uploaded_file(
            uploaded_meta, default_vpc, signal_reliability, scale_pct, simulations, profile
        )
        channel_outputs["Meta Ads"] = out
        platform_key, df_raw, import_result, result, decision = out

        # Only include in portfolio if not blocked
        if decision.get("status") != "DECISION_BLOCKED":
            channels_for_portfolio["meta"] = build_channel_decision("Meta Ads", import_result, result, decision)

    except Exception as e:
        st.error(f"Meta file error: {e}")

# Process Google
if uploaded_google:
    try:
        out = process_uploaded_file(
            uploaded_google, default_vpc, signal_reliability, scale_pct, simulations, profile
        )
        channel_outputs["Google Ads"] = out
        platform_key, df_raw, import_result, result, decision = out

        # Only include in portfolio if not blocked
        if decision.get("status") != "DECISION_BLOCKED":
            channels_for_portfolio["google"] = build_channel_decision("Google Ads", import_result, result, decision)

    except Exception as e:
        st.error(f"Google file error: {e}")

if not channel_outputs:
    st.error("No valid uploads could be processed. Please check your files.")
    st.stop()

# -----------------------------
# Per-channel output
# -----------------------------
for label, (platform_key, df_raw, import_result, result, decision) in channel_outputs.items():
    st.divider()
    st.header(f"{label}")
    st.caption(
        f"Engine: {result.get('engine_version', 'n/a')} | "
        f"Ruleset: {result.get('ruleset_version', 'n/a')} | "
        f"Seed: {result.get('random_seed', 'n/a')}"
    )

    st.subheader("Raw Preview (uploaded file)")
    st.dataframe(df_raw.head(10))

    # -----------------------------
    # Data Window (UI)
    # -----------------------------
    st.subheader("Data Window")

    date_min = result.get("date_min")
    date_max = result.get("date_max")
    days_of_data = result.get("days_of_data")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Days of data", days_of_data if days_of_data is not None else "n/a")
    with c2:
        if date_min and date_max:
            st.metric("Date range", f"{date_min} → {date_max}")
        else:
            st.metric("Date range", "n/a")

    # Validation info
    v = result.get("validation", {})
    if v:
        if v.get("status") == "DECISION_BLOCKED":
            st.error(f"Validation failed: {v.get('block_reason')}")
        else:
            st.success("Validation passed: data is suitable for decisioning.")

        for w in v.get("warnings", []):
            st.warning(w)

        with st.expander("Validation metrics"):
            st.json(v.get("metrics", {}))

    st.subheader("Detected column mapping")
    with st.expander("Detected column mapping"):
        st.json(import_result.detected_columns)

    for w in import_result.warnings:
        st.warning(w)

    st.subheader("Normalized Preview (engine-ready)")
    st.dataframe(import_result.df.head(10))
    # -----------------------------
    # Download normalized data (industry standard)
    # -----------------------------
    csv_bytes = import_result.df.to_csv(index=False).encode("utf-8")
    safe_label = label.lower().replace(" ", "_")
    st.download_button(
        label=f"Download Normalized CSV ({label})",
        data=csv_bytes,
        file_name=f"normalized_{safe_label}.csv",
        mime="text/csv",
        key=f"download_norm_{safe_label}",
    )

    # -----------------------------
    # Trends (Industry standard)
    # -----------------------------
    st.subheader("Trends (Daily)")

    df_plot = import_result.df.copy()
    df_plot["date"] = pd.to_datetime(df_plot["date"], errors="coerce")
    df_plot = df_plot.dropna(subset=["date"]).sort_values("date")

    # Spend trend
    st.write("**Spend over time**")
    st.line_chart(df_plot.set_index("date")[["spend"]])

    # Net value trend
    st.write("**Net value over time**")
    st.line_chart(df_plot.set_index("date")[["net_value"]])

    # -----------------------------
    # Decision
    # -----------------------------
    st.subheader("Decision")

    status = decision.get("status", "DECISION_OK")
    tier = decision.get("confidence_tier", "n/a")

    if status == "DECISION_BLOCKED":
        st.error(f"DECISION BLOCKED: {decision.get('reason')}")
    elif status == "DECISION_WARN":
        st.warning("Decision issued with caution.")
    else:
        st.success("Decision OK.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Action", decision.get("action", "n/a"))
    with c2:
        st.metric("Confidence Tier", tier)
    with c3:
        st.metric("Budget change", decision.get("recommended_change_pct_range", "n/a"))

    st.write("**Reason:**", decision.get("reason", ""))
    st.write("**User Explanation:**", decision.get("user_explanation", ""))
    st.caption(f"Next review: {decision.get('next_review_window', 'n/a')}")

    # Explainability
    exp = decision.get("explainability")
    if exp:
        st.subheader("Why this decision?")
        st.write("**What happened**")
        for x in exp.get("what_happened", []):
            st.write(f"- {x}")

        st.write("**What could go wrong**")
        for x in exp.get("what_could_go_wrong", []):
            st.write(f"- {x}")

        st.write("**What to do next**")
        for x in exp.get("what_to_do_next", []):
            st.write(f"- {x}")

    # -----------------------------
    # Summary
    # -----------------------------
    st.subheader("Summary")
    if decision.get("status") == "DECISION_BLOCKED":
        st.code(f"Decision blocked: {decision.get('reason')}")
    else:
        summary = recommendation_summary(result, decision, profile.name)
        st.code(summary)

    # -----------------------------
    # Report + Logging
    # -----------------------------
    if st.button(f"Generate Report File ({label})", key=f"report_{label}"):

        # Ensure these exist for reporting
        result["spend_total"] = float(import_result.df["spend"].sum())
        result["simulations"] = simulations
        result["signal_reliability"] = signal_reliability
        result["scale_pct"] = scale_pct

        report_path = write_markdown_report(
            result,
            decision,
            profile.name,
            platform_label=label
        )

        st.success(f"Report saved: {report_path}")

        # Download immediately
        try:
            with open(report_path, "rb") as f:
                st.download_button(
                    label=f"Download Report ({label})",
                    data=f.read(),
                    file_name=report_path.split("/")[-1],
                    mime="text/markdown",
                    key=f"download_{label}",
                )
        except Exception as e:
            st.warning(f"Could not create download button: {e}")

        # Log only when report is generated
        log_decision({
            "type": "channel",
            "platform": label,
            "status": decision.get("status"),
            "action": decision.get("action"),
            "confidence_tier": decision.get("confidence_tier"),
            "confidence": float(result.get("decision_confidence", 0.0)),
            "downside_risk": float(result.get("downside_risk", 1.0)),
            "spend_total": float(result.get("spend_total", import_result.df["spend"].sum())),
            "days_of_data": int(result.get("days_of_data", 0)),
            "date_min": result.get("date_min"),
            "date_max": result.get("date_max"),
            "engine_version": result.get("engine_version"),
            "ruleset_version": result.get("ruleset_version"),
            "random_seed": int(result.get("random_seed") or 0),
            "recommended_change_pct_range": decision.get("recommended_change_pct_range"),
            "next_review_window": decision.get("next_review_window"),
            "validation_status": (result.get("validation", {}) or {}).get("status"),
            "block_reason": (result.get("validation", {}) or {}).get("block_reason"),
        })


# -----------------------------
# Portfolio decision
# -----------------------------
st.divider()
st.header("Portfolio Decision (Cross-channel)")
# -----------------------------
# Portfolio controls (industry standard)
# -----------------------------
st.subheader("Portfolio Controls")

pc1, pc2, pc3 = st.columns(3)
with pc1:
    min_portfolio_conf = st.slider(
        "Min portfolio confidence",
        0.0, 1.0, 0.60, 0.05,
        help="Portfolio reallocation only triggers if confidence meets this threshold."
    )
with pc2:
    min_signal_separation = st.number_input(
        "Min signal separation (avg net value delta)",
        min_value=0.0,
        value=500.0,
        step=50.0,
        help="Reallocation only triggers if winner vs loser avg net value differs by at least this amount."
    )
with pc3:
    max_allowed_downside_risk = st.slider(
        "Max allowed downside risk (to reallocate into)",
        0.0, 1.0, 0.55, 0.05,
        help="Don't reallocate into a channel if its downside risk is above this."
    )

if len(channels_for_portfolio) >= 2:
    try:
        portfolio = recommend_portfolio_action(
            channels_for_portfolio,
            risk_profile=profile_name,
            min_portfolio_confidence=min_portfolio_conf,
            min_signal_separation=min_signal_separation,
            max_allowed_downside_risk=max_allowed_downside_risk,
        )

        st.subheader("Recommended Action")
        st.write(portfolio.portfolio_action)
        st.metric("Portfolio Confidence", f"{portfolio.portfolio_confidence:.2f}")

        rec = portfolio.recommendation

        if rec.enabled:
            st.success(f"Reallocate {rec.amount:,.0f} from {rec.from_platform} → {rec.to_platform}")
            st.write(f"Expected downside risk reduction: {rec.expected_downside_risk_reduction_pct:.1f}%")
            st.write(f"Expected confidence change: {rec.expected_confidence_change:+.3f}")
        else:
            st.info("No reallocation recommended.")
            if getattr(rec, "rationale", None):
                for r in rec.rationale:
                    st.write(f"- {r}")

        st.subheader("Why this decision?")
        st.write("**What happened**")
        for x in portfolio.rationale_blocks.get("what_happened", []):
            st.write(f"- {x}")

        st.write("**What could go wrong**")
        for x in portfolio.rationale_blocks.get("what_could_go_wrong", []):
            st.write(f"- {x}")

        st.write("**What to do next**")
        for x in portfolio.rationale_blocks.get("what_to_do_next", []):
            st.write(f"- {x}")

        if st.button("Log Portfolio Decision", key="log_portfolio"):
            log_decision({
                "type": "portfolio",
                "platform": "Portfolio",
                "action": portfolio.portfolio_action,
                "portfolio_confidence": float(portfolio.portfolio_confidence),
                "reallocation_enabled": bool(rec.enabled),
                "from_platform": rec.from_platform if rec.enabled else None,
                "to_platform": rec.to_platform if rec.enabled else None,
                "amount": float(rec.amount) if rec.enabled else 0.0,
                "engine_version": ENGINE_VERSION,
                "ruleset_version": RULESET_VERSION,
            })
            st.success("Portfolio decision logged to history.")

    except Exception as e:
        st.error(f"Portfolio decision error: {e}")

else:
    st.info("Upload both Meta and Google files (and pass validation) to get a portfolio recommendation.")

# -----------------------------
# Decision history (always at bottom)
# -----------------------------
st.divider()
st.header("Decision History (last 10 runs)")

history = read_history(limit=10)

if not history:
    st.info("No decision history yet. Generate a report to create history.")
else:
    dfh = pd.DataFrame(history).fillna("")

    # Put most important columns first (only if they exist)
    preferred = [
        "logged_at_utc", "type", "platform", "status", "action",
        "confidence_tier", "confidence", "downside_risk",
        "days_of_data", "date_min", "date_max",
        "engine_version", "ruleset_version", "random_seed",
        "recommended_change_pct_range", "next_review_window",
        "validation_status", "block_reason",
    ]

    cols = [c for c in preferred if c in dfh.columns] + [
        c for c in dfh.columns if c not in preferred
    ]
    dfh = dfh[cols]

    # Download CSV
    csv_hist = dfh.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download History (CSV)",
        data=csv_hist,
        file_name="mdu_history.csv",
        mime="text/csv",
        key="download_history_csv",
    )

    # Show table
    st.dataframe(dfh, use_container_width=True)

st.divider()
st.caption(
    "© 2026 Satish Saka · MDU Engine · "
    "Open-source under MIT License · "
    "Decision-support tool (not financial advice)"
)
import streamlit as st
import io
import pandas as pd
import hashlib
import json
from datetime import datetime, timezone
from datetime import datetime

from mdu_engine.decision_confidence import compute_decision_confidence
from mdu_engine.decision_rules import RISK_PROFILES, decide_action
from mdu_engine.reporting import recommendation_summary, write_markdown_report
from mdu_engine.importers.router import route_import
from mdu_engine.history import log_decision, read_history, get_latest_decision
from mdu_engine.version import ENGINE_VERSION, RULESET_VERSION

from mdu_engine.validation import validate_normalized_daily_schema, validation_to_dict

# NEW (portfolio)
from mdu_engine.portfolio_decision import ChannelDecision, recommend_portfolio_action


# -----------------------------
# Config (Owner-grade defaults)
# -----------------------------
MIN_DAYS_RECOMMENDED = 7
MAX_DAYS_RECOMMENDED = 30
AUTO_LOG_RUNS = True  # silent behavioural evidence; avoids spam via per-hash gating


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
    cols = [c for c in ["date", "spend", "conversions", "value_per_conversion", "net_value"] if c in df2.columns]
    df2 = df2[cols].copy()

    df2["date"] = pd.to_datetime(df2["date"], errors="coerce").dt.date.astype(str)
    for c in cols:
        if c != "date":
            df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0.0).round(6)

    payload = df2.to_csv(index=False).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)


def input_hash_from_norm_df(df_norm: pd.DataFrame) -> str:
    """
    Stable hash used for anti-gaming / rerun detection and run logging.
    """
    df2 = df_norm.copy()
    cols = [c for c in ["date", "spend", "conversions", "value_per_conversion", "net_value"] if c in df2.columns]
    df2 = df2[cols].copy()

    df2["date"] = pd.to_datetime(df2["date"], errors="coerce").dt.date.astype(str)
    for c in cols:
        if c != "date":
            df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0.0).round(6)

    payload = df2.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# -----------------------------
# Decision explanation template (immutable)
# -----------------------------
def build_structured_explanation(
    decision_outcome: str,
    primary_constraint: str,
    supporting_factors: list[str] | None,
    confidence_level: str,
    operator_consideration: str,
) -> dict:
    return {
        "Decision Outcome": decision_outcome,
        "Primary Constraint": primary_constraint,
        "Supporting Factors": supporting_factors or [],
        "Confidence Level": confidence_level,
        "Operator Consideration": operator_consideration,
    }


def classify_primary_constraint(status: str, action: str, validation_status: str | None) -> str:
    """
    Conservative, deterministic mapping.
    (In v1: we infer from outcome; later you can push this into decide_action itself.)
    """
    if status == "DECISION_BLOCKED" or (validation_status == "DECISION_BLOCKED"):
        return "Data Sufficiency Gate"
    if action == "REDUCE":
        return "Downside Risk Gate"
    if action == "HOLD":
        return "Confidence Gate"
    if action == "SCALE":
        return "Confidence Gate"
    return "Confidence Gate"


def decision_mode_from_status_action(status: str, action: str) -> str:
    """
    Epistemic = BLOCK, Strategic = HOLD/REDUCE, Permissive = SCALE.
    """
    if status == "DECISION_BLOCKED":
        return "Epistemic (refusal under uncertainty)"
    if action in ("HOLD", "REDUCE"):
        return "Strategic (restraint under known risk)"
    if action == "SCALE":
        return "Permissive (action allowed under constraints)"
    return "Strategic (restraint under known risk)"

def validate_snapshot(snapshot: dict) -> tuple[bool, list[str], list[str]]:
    """
    Validates an MDU Engine audit snapshot for replay and governance checks.
    Returns (ok, errors, warnings).
    """
    errors = []
    warnings = []

    if not isinstance(snapshot, dict):
        return False, ["Snapshot is not a JSON object."], []

    required_fields = [
        "snapshot_type",
        "platform",
        "logged_at_utc",
        "engine_version",
        "ruleset_version",
        "random_seed",
        "simulations",
        "signal_reliability",
        "scale_pct",
        "days_of_data",
        "date_min",
        "date_max",
        "validation",
        "decision",
    ]

    for f in required_fields:
        if f not in snapshot:
            errors.append(f"Missing required field: {f}")

    if errors:
        return False, errors, warnings

    # Type / sanity checks
    if not isinstance(snapshot["decision"], dict):
        errors.append("decision must be an object.")
    if not isinstance(snapshot["validation"], dict):
        errors.append("validation must be an object.")

    try:
        seed = int(snapshot["random_seed"])
        if seed <= 0:
            warnings.append("random_seed <= 0 (should normally be deterministic and > 0).")
    except Exception:
        errors.append("random_seed must be an integer.")

    # Version governance
    if snapshot["engine_version"] != ENGINE_VERSION:
        warnings.append(
            f"Engine version mismatch: snapshot={snapshot['engine_version']} current={ENGINE_VERSION}"
        )
    if snapshot["ruleset_version"] != RULESET_VERSION:
        warnings.append(
            f"Ruleset version mismatch: snapshot={snapshot['ruleset_version']} current={RULESET_VERSION}"
        )

    # Decision schema sanity
    for k in ["action", "status", "confidence_tier"]:
        if k not in snapshot["decision"]:
            warnings.append(f"Decision missing field: {k}")
    # ✅ Anti-tamper: verify snapshot_hash integrity
    expected_hash = snapshot.get("snapshot_hash")
    if not expected_hash:
        warnings.append("Missing snapshot_hash (anti-tamper check not available).")
    else:
        snap_copy = dict(snapshot)
        snap_copy.pop("snapshot_hash", None)
        calc = hashlib.sha256(json.dumps(snap_copy, sort_keys=True).encode("utf-8")).hexdigest()
        if calc != expected_hash:
            errors.append("snapshot_hash mismatch (snapshot may be edited/tampered).")

    ok = len(errors) == 0
    return ok, errors, warnings
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
    """
    Fail-closed pipeline:
    - Any validation failure => DECISION_BLOCKED (no MC)
    - Any unexpected exception => DECISION_BLOCKED (safety)
    """
    try:
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
            # placeholders
            "decision_confidence": 0.0,
            "downside_risk": 1.0,
            "avg_net_value": float(df_norm["net_value"].mean()) if "net_value" in df_norm and len(df_norm) else 0.0,
        }

        # Fail-closed: blocked => no MC
        if not v.is_valid:
            decision = {
                "status": "DECISION_BLOCKED",
                "confidence_tier": "n/a",
                # action can be HOLD but should never be treated as a valid HOLD; status dominates.
                "action": "HOLD",
                "recommended_change_pct_range": "0% (no change)",
                "next_review_window": "After fixing export",
                "reason": v.block_reason,
                "user_explanation": (
                    "The export or data window is not suitable for a defensible decision. "
                    "This is a safety outcome, not an error."
                ),
            }

            decision["primary_constraint"] = "Data Sufficiency Gate"
            decision["decision_mode"] = decision_mode_from_status_action(decision["status"], decision["action"])
            decision["structured_explanation"] = build_structured_explanation(
                decision_outcome="BLOCK",
                primary_constraint=decision["primary_constraint"],
                supporting_factors=[v.block_reason] if v.block_reason else [],
                confidence_level="n/a",
                operator_consideration=(
                    f"Export a daily report with at least {MIN_DAYS_RECOMMENDED} days "
                    f"({MIN_DAYS_RECOMMENDED}–{MAX_DAYS_RECOMMENDED} recommended), then re-run."
                ),
            )

            # legacy explainability (kept)
            decision["explainability"] = {
                "what_happened": [v.block_reason],
                "what_could_go_wrong": [
                    "Acting on insufficient or malformed data can lead to incorrect budget changes."
                ],
                "what_to_do_next": [
                    f"Export a daily report (Breakdown: Day) with at least {MIN_DAYS_RECOMMENDED} days "
                    f"({MIN_DAYS_RECOMMENDED}–{MAX_DAYS_RECOMMENDED} recommended) and re-upload."
                ],
            }
            # For anti-gaming / logging
            decision["input_hash"] = input_hash_from_norm_df(df_norm)
            result["random_seed"] = stable_seed_from_df(df_norm)

            return routed.platform, df_raw, import_result, result, decision

        # ✅ Monte Carlo only if validation passed
        seed = stable_seed_from_df(df_norm)
        result["random_seed"] = seed
        result["input_hash"] = input_hash_from_norm_df(df_norm)

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

        # Attach governance fields (deterministic, non-marketing)
        validation_status = (result.get("validation", {}) or {}).get("status")
        decision["primary_constraint"] = classify_primary_constraint(
            status=decision.get("status", "DECISION_OK"),
            action=decision.get("action", "HOLD"),
            validation_status=validation_status
        )
        decision["decision_mode"] = decision_mode_from_status_action(
            decision.get("status", "DECISION_OK"),
            decision.get("action", "HOLD"),
        )

        # Confidence label (no pseudo precision)
        conf = float(result.get("decision_confidence", 0.0))
        if decision.get("status") == "DECISION_BLOCKED":
            conf_level = "n/a"
        elif conf >= 0.75:
            conf_level = "High"
        elif conf >= 0.50:
            conf_level = "Medium"
        else:
            conf_level = "Low"

        # Supporting factors (keep boring)
        supporting = []
        if "warnings" in (result.get("validation", {}) or {}):
            supporting.extend((result.get("validation", {}) or {}).get("warnings", []) or [])
        if decision.get("reason"):
            supporting.append(decision.get("reason"))

        outcome = "BLOCK" if decision.get("status") == "DECISION_BLOCKED" else decision.get("action", "HOLD")

        decision["structured_explanation"] = build_structured_explanation(
            decision_outcome=outcome,
            primary_constraint=decision.get("primary_constraint", "Confidence Gate"),
            supporting_factors=supporting[:6],  # limit noise
            confidence_level=conf_level,
            operator_consideration=(
                "Treat HOLD and BLOCK as valid outcomes. Re-run only when conditions materially change "
                "(more data, improved stability, or reduced downside exposure)."
            ),
        )

        decision["input_hash"] = result.get("input_hash")

        return routed.platform, df_raw, import_result, result, decision

    except Exception as e:
        # HARD FAIL-CLOSED: never “error out” into a confusing state
        # Return a blocked decision with minimal safe structure.
        decision = {
            "status": "DECISION_BLOCKED",
            "confidence_tier": "n/a",
            "action": "HOLD",
            "recommended_change_pct_range": "0% (no change)",
            "next_review_window": "After fixing export",
            "reason": f"Processing failed safely: {str(e)}",
            "user_explanation": (
                "The system could not process the file into a defensible daily decision dataset. "
                "This is treated as a safety block."
            ),
            "primary_constraint": "Data Sufficiency Gate",
            "decision_mode": "Epistemic (refusal under uncertainty)",
            "structured_explanation": build_structured_explanation(
                decision_outcome="BLOCK",
                primary_constraint="Data Sufficiency Gate",
                supporting_factors=[f"Processing failure: {str(e)}"],
                confidence_level="n/a",
                operator_consideration="Verify the export format (daily breakdown) and re-upload.",
            ),
        }
        # Return minimal placeholders for UI
        df_raw = pd.DataFrame()
        import_result = type("ImportResult", (), {"df": pd.DataFrame(), "warnings": [], "detected_columns": {}})()
        result = {
            "date_min": None,
            "date_max": None,
            "days_of_data": 0,
            "spend_total": 0.0,
            "validation": {"status": "DECISION_BLOCKED", "block_reason": "Processing failure"},
            "engine_version": ENGINE_VERSION,
            "ruleset_version": RULESET_VERSION,
            "decision_confidence": 0.0,
            "downside_risk": 1.0,
            "avg_net_value": 0.0,
            "random_seed": 0,
            "input_hash": None,
        }
        return "unknown", df_raw, import_result, result, decision


# -----------------------------
# Build ChannelDecision for portfolio
# -----------------------------
def build_channel_decision(label: str, import_result, result: dict, decision: dict) -> ChannelDecision:
    df_norm = import_result.df
    spend_total = float(result.get("spend_total", df_norm["spend"].sum() if "spend" in df_norm else 0.0))
    avg_net_value = float(result.get("avg_net_value", df_norm["net_value"].mean() if "net_value" in df_norm else 0.0))

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
        notes=tuple(getattr(import_result, "warnings", []) or []),
    )


# -----------------------------
# Streamlit UI (Industry-grade)
# -----------------------------
st.set_page_config(page_title="MDU Engine", layout="wide")

# Guardrail: intended use (hostile-environment safe)
st.warning(
    "Important — Intended Use\n\n"
    "MDU Engine is designed for experienced decision-makers evaluating real advertising spend decisions.\n\n"
    "It is not intended for experimentation, optimisation, or exploratory testing.",
    icon="⚠️"
)

# Responsibility acknowledgement: once per session, hard gate
if "ack_responsibility" not in st.session_state:
    st.session_state.ack_responsibility = False

with st.expander("Responsibility acknowledgement (required)", expanded=True):
    st.markdown(
        "**Before you view any decision output:**\n\n"
        "MDU Engine provides guidance only. It does not execute changes automatically.\n\n"
        "You remain fully responsible for any actions taken based on this output."
    )
    st.session_state.ack_responsibility = st.checkbox(
        "I understand this output is advisory and that I remain fully responsible for any actions taken.",
        value=st.session_state.ack_responsibility
    )

if not st.session_state.ack_responsibility:
    st.info("Please acknowledge responsibility to proceed.")
    st.stop()

st.info(
    "MDU Engine provides decision support from uploaded performance exports. "
    "It does not execute changes automatically. Apply human judgment."
)

st.title("MDU Engine")
st.markdown(
    """
**MDU Engine helps you decide _when not to act_ by making risk and uncertainty visible.**  
It is a **decision-support** system (not auto-execution). It issues a **Decision Outcome**
(**SCALE / HOLD / REDUCE / BLOCK**) only when conditions are suitable for defensible action.
"""
)

with st.expander("How to use MDU Engine (recommended flow)", expanded=True):
    st.markdown(
        f"""
1. Upload **{MIN_DAYS_RECOMMENDED}–{MAX_DAYS_RECOMMENDED} days** of clean daily data (Meta or Google exports).  
2. Review the **Decision Outcome** and **Primary Constraint**.  
3. Treat **BLOCK** and **HOLD** as valid outcomes.  
4. Re-run **only** when conditions materially change (more data, improved stability, reduced downside exposure).
"""
    )

with st.expander("How to interpret outcomes (important)", expanded=True):
    st.markdown(
        """
### HOLD is a valid outcome (not a failure)
HOLD means the current data does **not justify irreversible budget changes**.  
This is a protective outcome used when signals are noisy, volatile, or uncertain.

### BLOCK is a safety outcome
BLOCK means the export or data window is not suitable for defensible decisioning (e.g., too few days, malformed report).  
MDU Engine blocks unsafe decisions instead of guessing.
"""
    )

with st.expander("About / Status", expanded=False):
    st.write("MDU Engine — Decision Engine for Meta Ads + Google Ads")
    st.write(f"Engine Version: {ENGINE_VERSION}")
    st.write(f"Ruleset Version: {RULESET_VERSION}")
    st.write("Status: Running ✅")

with st.expander("✅ Upload Instructions (V1 formats only)", expanded=True):
    st.markdown(f"""
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
- Recommended window: **{MIN_DAYS_RECOMMENDED}–{MAX_DAYS_RECOMMENDED} days**
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

st.divider()
st.caption("© 2026 Satish Saka · MDU Engine · MIT License · Public Decision Engine")

if not uploaded_meta and not uploaded_google:
    st.info("Upload at least one CSV (Meta and/or Google) to proceed.")
    st.stop()

profile = RISK_PROFILES[profile_name]

# Session-level rerun detection + run logging de-dupe
if "last_input_hash_by_channel" not in st.session_state:
    st.session_state.last_input_hash_by_channel = {}
if "logged_run_hashes" not in st.session_state:
    st.session_state.logged_run_hashes = set()

channel_outputs = {}
channels_for_portfolio = {}

# Process Meta
if uploaded_meta:
    out = process_uploaded_file(
        uploaded_meta, default_vpc, signal_reliability, scale_pct, simulations, profile
    )
    channel_outputs["Meta Ads"] = out

    platform_key, df_raw, import_result, result, decision = out
    if decision.get("status") != "DECISION_BLOCKED":
        channels_for_portfolio["meta"] = build_channel_decision("Meta Ads", import_result, result, decision)

# Process Google
if uploaded_google:
    out = process_uploaded_file(
        uploaded_google, default_vpc, signal_reliability, scale_pct, simulations, profile
    )
    channel_outputs["Google Ads"] = out

    platform_key, df_raw, import_result, result, decision = out
    if decision.get("status") != "DECISION_BLOCKED":
        channels_for_portfolio["google"] = build_channel_decision("Google Ads", import_result, result, decision)

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

    # Anti-gaming: warn on identical input reruns
    ih = decision.get("input_hash") or result.get("input_hash")
    prev_hash = st.session_state.last_input_hash_by_channel.get(label)
    if ih and prev_hash and ih == prev_hash:
        st.warning(
            "Identical input detected. Re-running without material change will not alter the outcome.",
            icon="⚠️"
        )
    if ih:
        st.session_state.last_input_hash_by_channel[label] = ih

    # Silent run logging (behavioural evidence), de-duped by input hash
    if AUTO_LOG_RUNS and ih and (ih not in st.session_state.logged_run_hashes):
        st.session_state.logged_run_hashes.add(ih)
        try:
            log_decision({
                "type": "run",
                "platform": label,
                "logged_at_utc": utc_now_iso(),
                "status": decision.get("status"),
                "action": decision.get("action"),
                "decision_mode": decision.get("decision_mode"),
                "primary_constraint": decision.get("primary_constraint"),
                "confidence": float(result.get("decision_confidence", 0.0)),
                "downside_risk": float(result.get("downside_risk", 1.0)),
                "spend_total": float(result.get("spend_total", 0.0)),
                "days_of_data": int(result.get("days_of_data", 0)),
                "date_min": result.get("date_min"),
                "date_max": result.get("date_max"),
                "engine_version": result.get("engine_version"),
                "ruleset_version": result.get("ruleset_version"),
                "random_seed": int(result.get("random_seed") or 0),
                "validation_status": (result.get("validation", {}) or {}).get("status"),
                "block_reason": (result.get("validation", {}) or {}).get("block_reason"),
                "input_hash": ih,
            })
        except Exception:
            # never fail the UI for logging issues
            pass

    st.subheader("Raw Preview (uploaded file)")
    st.dataframe(df_raw.head(10))

    # -----------------------------
    # Data Window
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
        st.json(getattr(import_result, "detected_columns", {}))

    for w in getattr(import_result, "warnings", []) or []:
        st.warning(w)

    st.subheader("Normalized Preview (engine-ready)")
    st.dataframe(import_result.df.head(10))

    # Download normalized data
    csv_bytes = import_result.df.to_csv(index=False).encode("utf-8")
    safe_label = label.lower().replace(" ", "_")
    st.download_button(
        label=f"Download Normalized CSV ({label})",
        data=csv_bytes,
        file_name=f"normalized_{safe_label}.csv",
        mime="text/csv",
        key=f"download_norm_{safe_label}",
    )

    # Trends
    st.subheader("Trends (Daily)")
    df_plot = import_result.df.copy()
    if "date" in df_plot.columns:
        df_plot["date"] = pd.to_datetime(df_plot["date"], errors="coerce")
        df_plot = df_plot.dropna(subset=["date"]).sort_values("date")

        if "spend" in df_plot.columns:
            st.write("**Spend over time**")
            st.line_chart(df_plot.set_index("date")[["spend"]])

        if "net_value" in df_plot.columns:
            st.write("**Net value over time**")
            st.line_chart(df_plot.set_index("date")[["net_value"]])
    else:
        st.info("Trend charts unavailable: normalized data has no date column.")

    # -----------------------------
    # Decision (Structured & governance-forward)
    # -----------------------------
    st.subheader("Decision")

    status = decision.get("status", "DECISION_OK")
    tier = decision.get("confidence_tier", "n/a")
    action = decision.get("action", "HOLD")

    # Decision outcome label
    if status == "DECISION_BLOCKED":
        st.error("Decision Outcome: BLOCK (Safety Outcome)")
        st.write("**Why this was blocked**")
        st.write(
            "MDU Engine blocks decisions when conditions are unsuitable for informed action. "
            "This is a safety outcome, not an error."
        )
        st.write("**Blocking reason:**", decision.get("reason", ""))

        st.write("**What to do next:**")
        exp = decision.get("explainability", {}) or {}
        steps = exp.get("what_to_do_next", []) or []
        if steps:
            for s in steps:
                st.write(f"- {s}")
        else:
            st.write(f"- Export a daily report ({MIN_DAYS_RECOMMENDED}–{MAX_DAYS_RECOMMENDED} days recommended) and re-upload.")

    elif status == "DECISION_WARN":
        st.warning("Decision issued with caution.")
    else:
        st.success("Decision issued (non-executing).")

    # Metrics row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Decision Outcome", "BLOCK" if status == "DECISION_BLOCKED" else action)
    with c2:
        st.metric("Confidence Tier", tier)
    with c3:
        st.metric("Budget change", decision.get("recommended_change_pct_range", "n/a"))

    # Governance details
    st.write("**Decision Mode:**", decision.get("decision_mode", "n/a"))
    st.write("**Primary Constraint:**", decision.get("primary_constraint", "n/a"))
    st.write("**Reason (factual):**", decision.get("reason", ""))
    st.write("**User Explanation (non-prescriptive):**", decision.get("user_explanation", ""))

    # Structured explanation (immutable schema)
    st.subheader("Structured explanation (audit-friendly)")
    se = decision.get("structured_explanation", {}) or {}
    if se:
        st.write(f"**Decision Outcome:** {se.get('Decision Outcome', '')}")
        st.write(f"**Primary Constraint:** {se.get('Primary Constraint', '')}")

        st.write("**Supporting Factors:**")
        factors = se.get("Supporting Factors", []) or []
        if factors:
            for f in factors:
                st.write(f"- {f}")
        else:
            st.write("- n/a")

        st.write(f"**Confidence Level:** {se.get('Confidence Level', '')}")
        st.write(f"**Operator Consideration:** {se.get('Operator Consideration', '')}")
    else:
        st.info("Structured explanation unavailable.")

    st.caption(f"Next review: {decision.get('next_review_window', 'n/a')}")

    # Legacy explainability (kept for continuity)
    exp = decision.get("explainability")
    if exp:
        st.subheader("Why this outcome? (detail)")
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
    # Summary (keep existing but remove “recommendation” tone)
    # -----------------------------
    st.subheader("Summary")
    if status == "DECISION_BLOCKED":
        st.code(f"Outcome: BLOCK • Primary constraint: {decision.get('primary_constraint')} • Reason: {decision.get('reason')}")
    else:
        summary = recommendation_summary(result, decision, profile.name)
        st.code(summary)

    # -----------------------------
    # Report + Logging (existing behaviour: only when report generated)
    # -----------------------------
    if st.button(f"Generate Report File ({label})", key=f"report_{label}"):

        # Ensure these exist for reporting
        result["spend_total"] = float(import_result.df["spend"].sum())
        result["simulations"] = simulations
        result["signal_reliability"] = signal_reliability
        result["scale_pct"] = scale_pct

        # ✅ Ensure deterministic seed exists for audit replay
        if not result.get("random_seed"):
            result["random_seed"] = stable_seed_from_df(import_result.df)

        report_path = write_markdown_report(
            result,
            decision,
            profile.name,
            platform_label=label
        )

        st.success(f"Report saved: {report_path}")
    # -----------------------------
    # ✅ Enterprise audit snapshot (JSON)
    # -----------------------------
    snapshot = {
        "snapshot_type": "channel_decision",
        "platform": label,
        "logged_at_utc": datetime.utcnow().isoformat() + "Z",
        "engine_version": result.get("engine_version"),
        "ruleset_version": result.get("ruleset_version"),
        "random_seed": int(result.get("random_seed") or 0),
        "input_hash": input_hash_from_norm_df(import_result.df),
        "simulations": int(result.get("simulations") or 0),
        "signal_reliability": float(result.get("signal_reliability") or 0.0),
        "scale_pct": float(result.get("scale_pct") or 0.0),
        "days_of_data": int(result.get("days_of_data") or 0),
        "date_min": result.get("date_min"),
        "date_max": result.get("date_max"),
        "validation": result.get("validation", {}),
        "decision": decision,
    }
    # ✅ Anti-tamper: hash the snapshot itself (excluding snapshot_hash field)
    snapshot_for_hash = dict(snapshot)
    snapshot_for_hash.pop("snapshot_hash", None)

    snapshot_bytes_for_hash = json.dumps(snapshot_for_hash, sort_keys=True).encode("utf-8")
    snapshot["snapshot_hash"] = hashlib.sha256(snapshot_bytes_for_hash).hexdigest()

    snapshot_bytes = pd.Series(snapshot).to_json().encode("utf-8")

    st.download_button(
        label=f"Download Audit Snapshot (JSON) — {label}",
        data=snapshot_bytes,
        file_name=f"mdu_snapshot_{label.lower().replace(' ', '_')}.json",
        mime="application/json",
        key=f"download_snapshot_{label}",
    )

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

    # Log only when report is generated (your existing high-signal audit)
    log_decision({
            "type": "channel",
            "platform": label,
            "status": decision.get("status"),
            "action": decision.get("action"),
            "decision_mode": decision.get("decision_mode"),
            "primary_constraint": decision.get("primary_constraint"),
            "confidence_tier": decision.get("confidence_tier"),
            "confidence": float(result.get("decision_confidence", 0.0)),
            "downside_risk": float(result.get("downside_risk", 1.0)),
            "spend_total": float(result.get("spend_total", 0.0)),
            "days_of_data": int(result.get("days_of_data", 0)),
            "date_min": result.get("date_min"),
            "date_max": result.get("date_max"),
            "engine_version": result.get("engine_version"),
            "ruleset_version": result.get("ruleset_version"),
            "random_seed": int(result["random_seed"]),
            "recommended_change_pct_range": decision.get("recommended_change_pct_range"),
            "next_review_window": decision.get("next_review_window"),
            "validation_status": (result.get("validation", {}) or {}).get("status"),
            "block_reason": (result.get("validation", {}) or {}).get("block_reason"),
            "input_hash": decision.get("input_hash") or result.get("input_hash"),
    })


# -----------------------------
# Portfolio decision
# -----------------------------
st.divider()
st.header("Portfolio Decision (Cross-channel)")

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

        st.subheader("Portfolio Outcome (non-executing)")
        st.write(portfolio.portfolio_action)
        st.metric("Portfolio Confidence", f"{portfolio.portfolio_confidence:.2f}")

        rec = portfolio.recommendation

        if rec.enabled:
            st.success(f"Reallocate {rec.amount:,.0f} from {rec.from_platform} → {rec.to_platform}")
            st.write(f"Expected downside risk reduction: {rec.expected_downside_risk_reduction_pct:.1f}%")
            st.write(f"Expected confidence change: {rec.expected_confidence_change:+.3f}")
        else:
            st.info("No reallocation indicated.")
            if getattr(rec, "rationale", None):
                for r in rec.rationale:
                    st.write(f"- {r}")

        st.subheader("Rationale (audit-friendly)")
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
                "logged_at_utc": utc_now_iso(),
            })
            st.success("Portfolio decision logged to history.")

    except Exception as e:
        st.error(f"Portfolio decision error: {e}")
else:
    st.info("Upload both Meta and Google files (and pass validation) to get a portfolio outcome.")

# -----------------------------
# Snapshot Replay (Audit)
# -----------------------------
st.divider()
st.header("Replay a Snapshot (Audit)")

st.caption(
    "Upload an audit snapshot JSON previously generated by MDU Engine. "
    "This verifies structural integrity and governance metadata (versions, seed, settings)."
)

snapshot_file = st.file_uploader("Upload Snapshot JSON", type=["json"], key="snapshot_replay")

if snapshot_file:
    try:
        raw = snapshot_file.getvalue().decode("utf-8", errors="replace")
        snapshot = json.loads(raw)

        ok, errors, warnings = validate_snapshot(snapshot)

        st.subheader("Snapshot Preview")
        st.json(snapshot)

        st.subheader("Replay Result")
        if ok:
            st.success("REPLAY PASS ✅ — Snapshot is structurally valid and audit-ready.")
        else:
            st.error("REPLAY FAIL ❌ — Snapshot failed validation checks.")

        if warnings:
            st.warning("Warnings (non-fatal):")
            for w in warnings:
                st.write(f"- {w}")

        if errors:
            st.error("Errors (fatal):")
            for e in errors:
                st.write(f"- {e}")

        # Optional: log replay attempt to history (enterprise audit trail)
        if st.button("Log Replay Result", key="log_replay"):
            log_decision({
                "type": "replay",
                "platform": snapshot.get("platform", "unknown"),
                "replay_pass": bool(ok),
                "snapshot_type": snapshot.get("snapshot_type"),
                "engine_version": snapshot.get("engine_version"),
                "ruleset_version": snapshot.get("ruleset_version"),
                "random_seed": int(snapshot.get("random_seed") or 0),
                "simulations": int(snapshot.get("simulations") or 0),
                "signal_reliability": float(snapshot.get("signal_reliability") or 0.0),
                "scale_pct": float(snapshot.get("scale_pct") or 0.0),
                "days_of_data": int(snapshot.get("days_of_data") or 0),
                "date_min": snapshot.get("date_min"),
                "date_max": snapshot.get("date_max"),
                "logged_at_utc": datetime.utcnow().isoformat() + "Z",
            })
            st.success("Replay result logged to history.")

    except Exception as e:
        st.error(f"Could not read/parse snapshot JSON: {e}")
# -----------------------------
# Decision history (always at bottom)
# -----------------------------
st.divider()
st.header("Decision History (last 10 logs)")

history = read_history(limit=10)
latest = get_latest_decision()
if latest:
    st.subheader("Latest Decision (Audit Snapshot)")
    st.json(latest)

if not history:
    st.info("No history yet. Logs are created automatically per unique run, and when reports are generated.")
else:
    dfh = pd.DataFrame(history).fillna("")

    preferred = [
        "logged_at_utc", "type", "platform", "status", "action",
        "decision_mode", "primary_constraint",
        "confidence_tier", "confidence", "downside_risk",
        "days_of_data", "date_min", "date_max",
        "engine_version", "ruleset_version", "random_seed",
        "recommended_change_pct_range", "next_review_window",
        "validation_status", "block_reason", "input_hash",
    ]

    cols = [c for c in preferred if c in dfh.columns] + [c for c in dfh.columns if c not in preferred]
    dfh = dfh[cols]

    csv_hist = dfh.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download History (CSV)",
        data=csv_hist,
        file_name="mdu_history.csv",
        mime="text/csv",
        key="download_history_csv",
    )

    st.dataframe(dfh, use_container_width=True)

st.divider()
st.caption("© 2026 Satish Saka · MDU Engine · MIT License · Public Decision Engine")
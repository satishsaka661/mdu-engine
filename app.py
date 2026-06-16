import streamlit as st
import plotly.express as px
import io
import pandas as pd
import hashlib
import json
from datetime import datetime, timezone
import html

# PDF (ReportLab)
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm

from mdu_engine.decision_confidence import compute_decision_confidence
from mdu_engine.decision_rules import RISK_PROFILES, decide_action
from mdu_engine.reporting import recommendation_summary, write_markdown_report
from mdu_engine.importers.router import route_import
from mdu_engine.history import log_decision, read_history, get_latest_decision
from mdu_engine.version import ENGINE_VERSION, RULESET_VERSION
from mdu_engine.feedback import write_feedback, read_feedback, feedback_to_csv_bytes
from mdu_engine.validation import validate_normalized_daily_schema, validation_to_dict

# NEW (portfolio)
from mdu_engine.portfolio_decision import ChannelDecision, recommend_portfolio_action
from login_page import show_login_gate, show_user_header
from override_ui_section import show_override_log_prompt, show_followup_forms, show_override_dashboard


# -----------------------------
# Config (Owner-grade defaults)
# -----------------------------
MIN_DAYS_RECOMMENDED = 7
MAX_DAYS_RECOMMENDED = 30
AUTO_LOG_RUNS = True


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
        if (l.startswith("Day,") or l.startswith("Date,")) and ("Cost" in l) and ("Conversions" in l):
            header_idx = i
            sep = ","
            break
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


def build_audit_pdf_bytes(platform_label: str, risk_profile_name: str, result: dict, decision: dict, snapshot: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("<b>MDU Engine — Audit Report</b>", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"<b>Platform:</b> {platform_label}", styles["Normal"]))
    story.append(Paragraph(f"<b>Risk profile:</b> {risk_profile_name}", styles["Normal"]))
    story.append(Paragraph(f"<b>Generated (UTC):</b> {snapshot.get('logged_at_utc')}", styles["Normal"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Versions</b>", styles["Heading2"]))
    story.append(Paragraph(f"Engine: {result.get('engine_version')}  |  Ruleset: {result.get('ruleset_version')}", styles["Normal"]))
    story.append(Paragraph(f"Random seed: {result.get('random_seed')}  |  Input hash: {snapshot.get('input_hash')}", styles["Normal"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Data window</b>", styles["Heading2"]))
    story.append(Paragraph(f"Days of data: {result.get('days_of_data')}", styles["Normal"]))
    story.append(Paragraph(f"Date range: {result.get('date_min')} → {result.get('date_max')}", styles["Normal"]))
    story.append(Paragraph(f"Total spend: {result.get('spend_total')}", styles["Normal"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Decision</b>", styles["Heading2"]))
    story.append(Paragraph(f"Status: {decision.get('status')}", styles["Normal"]))
    story.append(Paragraph(f"Outcome: {'BLOCK' if decision.get('status')=='DECISION_BLOCKED' else decision.get('action')}", styles["Normal"]))
    story.append(Paragraph(f"Confidence tier: {decision.get('confidence_tier')}", styles["Normal"]))
    story.append(Paragraph(f"Recommended change: {decision.get('recommended_change_pct_range')}", styles["Normal"]))
    story.append(Paragraph(f"Primary constraint: {decision.get('primary_constraint')}", styles["Normal"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Reason</b>", styles["Heading2"]))
    story.append(Paragraph(str(decision.get("reason", "")), styles["Normal"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Validation</b>", styles["Heading2"]))
    v = result.get("validation", {}) or {}
    story.append(Paragraph(f"Validation status: {v.get('status')}", styles["Normal"]))
    if v.get("block_reason"):
        story.append(Paragraph(f"Block reason: {v.get('block_reason')}", styles["Normal"]))
    warnings = v.get("warnings", []) or []
    if warnings:
        story.append(Paragraph("Warnings:", styles["Normal"]))
        for w in warnings[:10]:
            story.append(Paragraph(f"- {w}", styles["Normal"]))
    doc.build(story)
    return buf.getvalue()


def report_text_to_pdf_bytes(title: str, md_text: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm, title=title)
    styles = getSampleStyleSheet()
    story = [Paragraph(html.escape(title), styles["Heading1"]), Spacer(1, 8)]
    for line in (md_text or "").splitlines():
        raw = line.strip()
        if not raw:
            story.append(Spacer(1, 6))
            continue
        if raw.startswith("### "):
            story.append(Paragraph(html.escape(raw[4:]), styles["Heading2"]))
            story.append(Spacer(1, 4))
        elif raw.startswith("## "):
            story.append(Paragraph(html.escape(raw[3:]), styles["Heading2"]))
            story.append(Spacer(1, 4))
        elif raw.startswith("# "):
            story.append(Paragraph(html.escape(raw[2:]), styles["Heading2"]))
            story.append(Spacer(1, 4))
        else:
            if raw.startswith("- "):
                raw = "• " + raw[2:]
            story.append(Paragraph(html.escape(raw), styles["BodyText"]))
    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def build_structured_explanation(decision_outcome, primary_constraint, supporting_factors, confidence_level, operator_consideration):
    return {
        "Decision Outcome": decision_outcome,
        "Primary Constraint": primary_constraint,
        "Supporting Factors": supporting_factors or [],
        "Confidence Level": confidence_level,
        "Operator Consideration": operator_consideration,
    }


def classify_primary_constraint(status, action, validation_status):
    if status == "DECISION_BLOCKED" or (validation_status == "DECISION_BLOCKED"):
        return "Data Sufficiency Gate"
    if action == "REDUCE":
        return "Downside Risk Gate"
    return "Confidence Gate"


def decision_mode_from_status_action(status, action):
    if status == "DECISION_BLOCKED":
        return "Epistemic (refusal under uncertainty)"
    if action in ("HOLD", "REDUCE"):
        return "Strategic (restraint under known risk)"
    if action == "SCALE":
        return "Permissive (action allowed under constraints)"
    return "Strategic (restraint under known risk)"


def validate_snapshot(snapshot):
    errors = []
    warnings = []
    if not isinstance(snapshot, dict):
        return False, ["Snapshot is not a JSON object."], []
    required_fields = ["snapshot_type", "platform", "logged_at_utc", "engine_version", "ruleset_version",
                       "random_seed", "simulations", "signal_reliability", "scale_pct", "days_of_data",
                       "date_min", "date_max", "validation", "decision"]
    for f in required_fields:
        if f not in snapshot:
            errors.append(f"Missing required field: {f}")
    if errors:
        return False, errors, warnings
    if not isinstance(snapshot["decision"], dict):
        errors.append("decision must be an object.")
    if not isinstance(snapshot["validation"], dict):
        errors.append("validation must be an object.")
    try:
        seed = int(snapshot["random_seed"])
        if seed <= 0:
            warnings.append("random_seed <= 0")
    except Exception:
        errors.append("random_seed must be an integer.")
    if snapshot["engine_version"] != ENGINE_VERSION:
        warnings.append(f"Engine version mismatch: snapshot={snapshot['engine_version']} current={ENGINE_VERSION}")
    if snapshot["ruleset_version"] != RULESET_VERSION:
        warnings.append(f"Ruleset version mismatch: snapshot={snapshot['ruleset_version']} current={RULESET_VERSION}")
    for k in ["action", "status", "confidence_tier"]:
        if k not in snapshot["decision"]:
            warnings.append(f"Decision missing field: {k}")
    expected_hash = snapshot.get("snapshot_hash")
    if not expected_hash:
        warnings.append("Missing snapshot_hash")
    else:
        snap_copy = dict(snapshot)
        snap_copy.pop("snapshot_hash", None)
        calc = hashlib.sha256(json.dumps(snap_copy, sort_keys=True).encode("utf-8")).hexdigest()
        if calc != expected_hash:
            errors.append("snapshot_hash mismatch (snapshot may be edited/tampered).")
    ok = len(errors) == 0
    return ok, errors, warnings


def process_uploaded_file(uploaded_file, default_vpc, signal_reliability, scale_pct, simulations, profile):
    try:
        df_raw = read_ads_csv(uploaded_file)
        routed = route_import(df_raw, default_value_per_conversion=default_vpc if default_vpc > 0 else None)
        import_result = routed.result
        df_norm = import_result.df
        v = validate_normalized_daily_schema(df_norm)
        v_dict = validation_to_dict(v)
        result = {
            "date_min": v.metrics.get("date_min"),
            "date_max": v.metrics.get("date_max"),
            "days_of_data": v.metrics.get("days_of_data", 0),
            "spend_total": float(v.metrics.get("spend_total", df_norm["spend"].sum() if "spend" in df_norm else 0.0)),
            "validation": v_dict,
            "engine_version": ENGINE_VERSION,
            "ruleset_version": RULESET_VERSION,
            "decision_confidence": 0.0,
            "downside_risk": 1.0,
            "avg_net_value": float(df_norm["net_value"].mean()) if "net_value" in df_norm and len(df_norm) else 0.0,
        }
        if not v.is_valid:
            decision = {
                "status": "DECISION_BLOCKED", "confidence_tier": "n/a", "action": "HOLD",
                "recommended_change_pct_range": "0% (no change)", "next_review_window": "After fixing export",
                "reason": v.block_reason,
                "user_explanation": "The export or data window is not suitable for a defensible decision. This is a safety outcome, not an error.",
            }
            decision["primary_constraint"] = "Data Sufficiency Gate"
            decision["decision_mode"] = decision_mode_from_status_action(decision["status"], decision["action"])
            decision["structured_explanation"] = build_structured_explanation(
                "BLOCK", decision["primary_constraint"], [v.block_reason] if v.block_reason else [], "n/a",
                f"Export a daily report with at least {MIN_DAYS_RECOMMENDED} days ({MIN_DAYS_RECOMMENDED}–{MAX_DAYS_RECOMMENDED} recommended), then re-run."
            )
            decision["explainability"] = {
                "what_happened": [v.block_reason],
                "what_could_go_wrong": ["Acting on insufficient or malformed data can lead to incorrect budget changes."],
                "what_to_do_next": [f"Export a daily report (Breakdown: Day) with at least {MIN_DAYS_RECOMMENDED} days and re-upload."],
            }
            decision["input_hash"] = input_hash_from_norm_df(df_norm)
            result["random_seed"] = stable_seed_from_df(df_norm)
            result["input_hash"] = decision["input_hash"]
            return routed.platform, df_raw, import_result, result, decision

        seed = stable_seed_from_df(df_norm)
        result["random_seed"] = seed
        result["input_hash"] = input_hash_from_norm_df(df_norm)
        mc = compute_decision_confidence(df_norm, signal_reliability=signal_reliability, scale_pct=scale_pct, simulations=simulations)
        result.update(mc)
        result["simulations"] = simulations
        result["signal_reliability"] = signal_reliability
        result["scale_pct"] = scale_pct
        decision = decide_action(decision_confidence=result["decision_confidence"], downside_risk=result["downside_risk"], profile=profile, days_of_data=result.get("days_of_data"))
        validation_status = (result.get("validation", {}) or {}).get("status")
        decision["primary_constraint"] = classify_primary_constraint(decision.get("status", "DECISION_OK"), decision.get("action", "HOLD"), validation_status)
        decision["decision_mode"] = decision_mode_from_status_action(decision.get("status", "DECISION_OK"), decision.get("action", "HOLD"))
        conf = float(result.get("decision_confidence", 0.0))
        if decision.get("status") == "DECISION_BLOCKED":
            conf_level = "n/a"
        elif conf >= 0.75:
            conf_level = "High"
        elif conf >= 0.50:
            conf_level = "Medium"
        else:
            conf_level = "Low"
        supporting = []
        supporting.extend((result.get("validation", {}) or {}).get("warnings", []) or [])
        if decision.get("reason"):
            supporting.append(decision.get("reason"))
        outcome = "BLOCK" if decision.get("status") == "DECISION_BLOCKED" else decision.get("action", "HOLD")
        decision["structured_explanation"] = build_structured_explanation(
            outcome, decision.get("primary_constraint", "Confidence Gate"), supporting[:6], conf_level,
            "Treat HOLD and BLOCK as valid outcomes. Re-run only when conditions materially change."
        )
        decision["input_hash"] = result.get("input_hash")
        return routed.platform, df_raw, import_result, result, decision

    except Exception as e:
        decision = {
            "status": "DECISION_BLOCKED", "confidence_tier": "n/a", "action": "HOLD",
            "recommended_change_pct_range": "0% (no change)", "next_review_window": "After fixing export",
            "reason": f"Processing failed safely: {str(e)}",
            "user_explanation": "The system could not process the file into a defensible daily decision dataset. This is treated as a safety block.",
            "primary_constraint": "Data Sufficiency Gate",
            "decision_mode": "Epistemic (refusal under uncertainty)",
            "structured_explanation": build_structured_explanation("BLOCK", "Data Sufficiency Gate", [f"Processing failure: {str(e)}"], "n/a", "Verify the export format (daily breakdown) and re-upload."),
        }
        df_raw = pd.DataFrame()
        import_result = type("ImportResult", (), {"df": pd.DataFrame(), "warnings": [], "detected_columns": {}})()
        result = {
            "date_min": None, "date_max": None, "days_of_data": 0, "spend_total": 0.0,
            "validation": {"status": "DECISION_BLOCKED", "block_reason": "Processing failure"},
            "engine_version": ENGINE_VERSION, "ruleset_version": RULESET_VERSION,
            "decision_confidence": 0.0, "downside_risk": 1.0, "avg_net_value": 0.0,
            "random_seed": 0, "input_hash": None,
        }
        return "unknown", df_raw, import_result, result, decision


def build_channel_decision(label, import_result, result, decision):
    df_norm = import_result.df
    spend_total = float(result.get("spend_total", df_norm["spend"].sum() if "spend" in df_norm else 0.0))
    avg_net_value = float(result.get("avg_net_value", df_norm["net_value"].mean() if "net_value" in df_norm else 0.0))
    action = decision.get("action", "HOLD")
    if action == "MAINTAIN":
        action = "HOLD"
    return ChannelDecision(
        platform=label, decision=action, avg_net_value=avg_net_value,
        downside_risk=float(result.get("downside_risk", 1.0)),
        confidence=float(result.get("decision_confidence", 0.0)),
        spend_total=spend_total, notes=tuple(getattr(import_result, "warnings", []) or []),
    )


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="MDU Engine", layout="wide")
if not show_login_gate():
    st.stop()
show_user_header()
show_followup_forms()

# Google Analytics
st.markdown("""
<script async src="https://www.googletagmanager.com/gtag/js?id=G-RM6RGSHRL0"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-RM6RGSHRL0');
</script>
""", unsafe_allow_html=True)

# ── Custom CSS Theme ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0D1B2A !important;
    color: #E8F0FE !important;
}

.main .block-container {
    padding: 1.5rem 2.5rem 3rem 2.5rem !important;
    max-width: 1200px !important;
    background-color: #0D1B2A !important;
}

header[data-testid="stHeader"] {
    background: linear-gradient(90deg, #0D1B2A 0%, #1A2E45 100%) !important;
    border-bottom: 2px solid #FF6B2B !important;
}

h1 {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #1E90FF 0%, #FF6B2B 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}

h2 {
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    color: #4AABFF !important;
    border-left: 3px solid #FF6B2B !important;
    padding-left: 0.6rem !important;
    margin-top: 1.5rem !important;
}

h3 { font-size: 1.05rem !important; font-weight: 600 !important; color: #FFB347 !important; }

hr { border-color: rgba(30,144,255,0.18) !important; }

[data-testid="stExpander"] {
    background-color: #1E3A5F !important;
    border: 1px solid rgba(30,144,255,0.18) !important;
    border-radius: 10px !important;
    margin-bottom: 0.75rem !important;
}

[data-testid="stExpander"] summary {
    background-color: #1A2E45 !important;
    color: #E8F0FE !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
}

[data-testid="stExpander"] summary:hover {
    background-color: #1E3A5F !important;
    color: #4AABFF !important;
}

[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary div,
[data-testid="stExpander"] summary svg {
    color: #E8F0FE !important;
    fill: #E8F0FE !important;
}

details > summary {
    background: #1A2E45 !important;
    color: #E8F0FE !important;
    list-style: none !important;
}

details > summary::marker,
details > summary::-webkit-details-marker {
    color: #4AABFF !important;
}

[data-testid="stMetric"] {
    background: #1E3A5F !important;
    border: 1px solid rgba(30,144,255,0.18) !important;
    border-radius: 10px !important;
    padding: 1rem 1.25rem !important;
}

[data-testid="stMetricLabel"] {
    color: #8BA3C7 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

[data-testid="stMetricValue"] {
    color: #4AABFF !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

[data-testid="stButton"] > button {
    background: linear-gradient(90deg, #1E90FF 0%, #FF6B2B 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

[data-testid="stDownloadButton"] > button {
    background: #1E3A5F !important;
    color: #4AABFF !important;
    border: 1px solid #1E90FF !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}

[data-testid="stFileUploader"] {
    background: #1E3A5F !important;
    border: 2px dashed #1E90FF !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #1A2E45 !important;
    border-radius: 8px !important;
}

[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] div {
    color: #8BA3C7 !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #4AABFF !important;
}

[data-testid="stFileUploader"] label {
    color: #E8F0FE !important;
    font-weight: 500 !important;
}

/* Uploaded file name */
[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p,
[data-testid="baseButton-minimal"] {
    color: #E8F0FE !important;
}

/* Browse files button */
[data-testid="stFileUploaderDropzone"] button {
    background: #1E3A5F !important;
    color: #4AABFF !important;
    border: 1px solid #1E90FF !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}

/* File name and delete button after upload */
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFile"] span,
[data-testid="stFileUploaderFile"] p,
[data-testid="stFileUploaderFile"] div,
[data-testid="stFileUploader"] [class*="uploadedFile"],
[data-testid="stFileUploader"] [class*="uploadedFile"] span,
[data-testid="stFileUploader"] [class*="uploadedFile"] p,
[data-testid="stFileUploader"] [class*="fileName"],
[data-testid="stFileUploader"] [class*="fileSize"] {
    color: #E8F0FE !important;
}

/* File icon after upload */
[data-testid="stFileUploaderFile"] svg {
    fill: #4AABFF !important;
    color: #4AABFF !important;
}

/* Delete file button */
[data-testid="stFileUploader"] button[kind="minimal"],
[data-testid="stFileUploader"] button[aria-label*="Delete"] {
    color: #FF6B2B !important;
    background: transparent !important;
    border: none !important;
}

/* File section background */
[data-testid="stFileUploader"] > div:last-child {
    background: #1A2E45 !important;
    border-radius: 6px !important;
    padding: 4px 8px !important;
}

/* Upload icon colour */
[data-testid="stFileUploaderDropzone"] svg {
    fill: #4AABFF !important;
    color: #4AABFF !important;
}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input {
    background: #1A2E45 !important;
    color: #E8F0FE !important;
    border: 1px solid rgba(30,144,255,0.18) !important;
    border-radius: 7px !important;
}

[data-testid="stSelectbox"] > div > div {
    background: #1A2E45 !important;
    border: 1px solid rgba(30,144,255,0.18) !important;
    border-radius: 7px !important;
    color: #E8F0FE !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(30,144,255,0.18) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

[data-testid="stCode"] {
    background: #1A2E45 !important;
    border: 1px solid rgba(30,144,255,0.18) !important;
    border-radius: 8px !important;
    color: #FFB347 !important;
}

[data-testid="stSidebar"] {
    background: #1A2E45 !important;
    border-right: 1px solid rgba(30,144,255,0.18) !important;
}

[data-testid="stJson"] {
    background: #1A2E45 !important;
    border-radius: 8px !important;
    border: 1px solid rgba(30,144,255,0.18) !important;
}

[data-testid="stCaptionContainer"] { color: #8BA3C7 !important; }

[data-testid="stForm"] [data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(90deg, #FF6B2B 0%, #1E90FF 100%) !important;
    font-size: 1rem !important;
    padding: 0.65rem 1.5rem !important;
}
[data-testid="stExpander"] details {
    background-color: #1E3A5F !important;
}

[data-testid="stExpander"] details > div {
    color: #E8F0FE !important;
}

section[data-testid="stMain"] {
    background-color: #0D1B2A !important;
}

.stMarkdown p, .stMarkdown li {
    color: #E8F0FE !important;
}

label[data-baseweb="checkbox"] span {
    color: #E8F0FE !important;
}
            /* Fix all label and widget text visibility */
label, .stTextInput label, .stTextArea label,
.stNumberInput label, .stSelectbox label,
.stSlider label, .stRadio label,
[data-testid="stWidgetLabel"] {
    color: #E8F0FE !important;
}

/* Fix expander inner text */
[data-testid="stExpander"] p,
[data-testid="stExpander"] span,
[data-testid="stExpander"] li,
[data-testid="stExpander"] div {
    color: #E8F0FE !important;
}

/* Fix file uploader inner box */
[data-testid="stFileUploaderDropzoneInstructions"] {
    background: #1A2E45 !important;
    color: #E8F0FE !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: #1A2E45 !important;
}

/* Fix form inner text */
[data-testid="stForm"] label,
[data-testid="stForm"] p,
[data-testid="stForm"] span {
    color: #E8F0FE !important;
}

/* Fix radio button labels */
[data-testid="stRadio"] span {
    color: #E8F0FE !important;
}

/* Fix placeholder text */
::placeholder {
    color: #5A7A9F !important;
    opacity: 1 !important;
}

/* Fix slider value text */
[data-testid="stSlider"] p {
    color: #E8F0FE !important;
}

/* Fix caption text globally */
p, span, div {
    color: inherit;
}
/* Fix chart backgrounds */
[data-testid="stArrowVegaLiteChart"] canvas,
.vega-embed canvas,
iframe {
    background-color: #0D1B2A !important;
    border-radius: 8px !important;
}
[data-testid="stArrowVegaLiteChart"] {
    background-color: #1E3A5F !important;
    border-radius: 8px !important;
    padding: 0.5rem !important;
}           
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0D1B2A; }
::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #1E90FF; }
</style>
""", unsafe_allow_html=True)

# ── Logo Header ───────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:1rem;padding:0.5rem 0 1.25rem 0;border-bottom:1px solid rgba(30,144,255,0.2);margin-bottom:1.5rem;">
    <div style="font-size:2rem;font-weight:800;background:linear-gradient(90deg,#1E90FF 0%,#FF6B2B 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-0.5px;">MDU Engine</div>
    <div style="font-size:0.78rem;color:#8BA3C7;border-left:1px solid rgba(30,144,255,0.3);padding-left:1rem;line-height:1.6;">
        Decision Systems &nbsp;·&nbsp; Data-Driven Optimisation<br>
        <span style="color:#FF6B2B;font-weight:600;font-size:0.82rem;">SCALE &nbsp;/&nbsp; HOLD &nbsp;/&nbsp; REDUCE &nbsp;/&nbsp; BLOCK</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Responsibility gate ───────────────────────────────────
st.warning(
    "Important — Intended Use\n\n"
    "MDU Engine is designed for experienced decision-makers evaluating real advertising spend decisions.\n\n"
    "It is not intended for experimentation, optimisation, or exploratory testing.",
    icon="⚠️"
)

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

st.info("MDU Engine provides decision support from uploaded performance exports. It does not execute changes automatically. Apply human judgment.")

st.markdown("""
**MDU Engine helps you decide _when not to act_ by making risk and uncertainty visible.**  
It is a **decision-support** system (not auto-execution). It issues a **Decision Outcome**
(**SCALE / HOLD / REDUCE / BLOCK**) only when conditions are suitable for defensible action.
""")

with st.expander("How to use MDU Engine (recommended flow)", expanded=True):
    st.markdown(f"""
1. Upload **{MIN_DAYS_RECOMMENDED}–{MAX_DAYS_RECOMMENDED} days** of clean daily data (Meta or Google exports).  
2. Review the **Decision Outcome** and **Primary Constraint**.  
3. Treat **BLOCK** and **HOLD** as valid outcomes.  
4. Re-run **only** when conditions materially change.
""")

with st.expander("How to interpret outcomes (important)", expanded=True):
    st.markdown("""
### HOLD is a valid outcome (not a failure)
HOLD means the current data does **not justify irreversible budget changes**.

### BLOCK is a safety outcome
BLOCK means the export or data window is not suitable for defensible decisioning.
""")

with st.expander("About / Status", expanded=False):
    st.write("MDU Engine — Decision Engine for Meta Ads + Google Ads")
    st.write(f"Engine Version: {ENGINE_VERSION}")
    st.write(f"Ruleset Version: {RULESET_VERSION}")
    st.write("Status: Running ✅")

with st.expander("✅ Upload Instructions (V1 formats only)", expanded=True):
    st.markdown(f"""
### Meta Ads (Daily Performance Export)
- Breakdown by **Day**
- Columns: `Date` / `Day` / `Reporting starts`, `Amount spent`, `Results`

### Google Ads (Campaign Performance Report)
- Report segmented by **Day**
- Columns: `Date` / `Day`, `Cost`, `Conversions` or `All conversions`

### General Rules
- One row per day · Mixed platforms in one CSV ❌
- Totals rows ignored automatically
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
default_vpc = st.number_input("Default Value per Conversion", min_value=0.0, value=5000.0, step=100.0)

st.divider()
st.caption("© 2026 Satish Saka · MDU Engine · MIT License · Public Decision Engine")

if not uploaded_meta and not uploaded_google:
    st.info("Upload at least one CSV (Meta and/or Google) to proceed.")
    st.stop()

profile = RISK_PROFILES[profile_name]

if "last_input_hash_by_channel" not in st.session_state:
    st.session_state.last_input_hash_by_channel = {}
if "logged_run_hashes" not in st.session_state:
    st.session_state.logged_run_hashes = set()

channel_outputs = {}
channels_for_portfolio = {}

if uploaded_meta:
    out = process_uploaded_file(uploaded_meta, default_vpc, signal_reliability, scale_pct, simulations, profile)
    channel_outputs["Meta Ads"] = out
    _, _, import_result, result, decision = out
    if decision.get("status") != "DECISION_BLOCKED":
        channels_for_portfolio["meta"] = build_channel_decision("Meta Ads", import_result, result, decision)

if uploaded_google:
    out = process_uploaded_file(uploaded_google, default_vpc, signal_reliability, scale_pct, simulations, profile)
    channel_outputs["Google Ads"] = out
    _, _, import_result, result, decision = out
    if decision.get("status") != "DECISION_BLOCKED":
        channels_for_portfolio["google"] = build_channel_decision("Google Ads", import_result, result, decision)

if not channel_outputs:
    st.error("No valid uploads could be processed. Please check your files.")
    st.stop()

# ── Per-channel output ────────────────────────────────────
for label, (platform_key, df_raw, import_result, result, decision) in channel_outputs.items():
    st.divider()
    st.header(f"{label}")
    st.caption(f"Engine: {result.get('engine_version', 'n/a')} | Ruleset: {result.get('ruleset_version', 'n/a')} | Seed: {result.get('random_seed', 'n/a')}")

    ih = decision.get("input_hash") or result.get("input_hash")
    prev_hash = st.session_state.last_input_hash_by_channel.get(label)
    if ih and prev_hash and ih == prev_hash:
        st.warning("Identical input detected. Re-running without material change will not alter the outcome.", icon="⚠️")
    if ih:
        st.session_state.last_input_hash_by_channel[label] = ih

    if AUTO_LOG_RUNS and ih and (ih not in st.session_state.logged_run_hashes):
        st.session_state.logged_run_hashes.add(ih)
        try:
            log_decision({
                "type": "run", "platform": label, "logged_at_utc": utc_now_iso(),
                "status": decision.get("status"), "action": decision.get("action"),
                "decision_mode": decision.get("decision_mode"), "primary_constraint": decision.get("primary_constraint"),
                "confidence": float(result.get("decision_confidence", 0.0)), "downside_risk": float(result.get("downside_risk", 1.0)),
                "spend_total": float(result.get("spend_total", 0.0)), "days_of_data": int(result.get("days_of_data", 0)),
                "date_min": result.get("date_min"), "date_max": result.get("date_max"),
                "engine_version": result.get("engine_version"), "ruleset_version": result.get("ruleset_version"),
                "random_seed": int(result.get("random_seed") or 0),
                "validation_status": (result.get("validation", {}) or {}).get("status"),
                "block_reason": (result.get("validation", {}) or {}).get("block_reason"),
                "input_hash": ih,
            })
        except Exception:
            pass

    st.subheader("Raw Preview (uploaded file)")
    st.dataframe(df_raw.head(10))

    st.subheader("Data Window")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Days of data", result.get("days_of_data", "n/a"))
    with c2:
        date_min, date_max = result.get("date_min"), result.get("date_max")
        st.metric("Date range", f"{date_min} → {date_max}" if date_min and date_max else "n/a")

    v = result.get("validation", {})
    if v:
        if v.get("status") == "DECISION_BLOCKED":
            st.error(f"Validation failed: {v.get('block_reason')}")
        else:
            st.success("Validation passed: data is suitable for decisioning.")
        for w in v.get("warnings", []):
            st.warning(w)
        with st.expander("Validation metrics"):
            st.markdown(f'<pre style="background:#1A2E45;color:#FFB347;padding:1rem;border-radius:8px;font-size:0.8rem;overflow-x:auto;white-space:pre;">{json.dumps(v.get("metrics", {}), indent=2)}</pre>', unsafe_allow_html=True)

    with st.expander("Detected column mapping"):
        st.markdown(f'<pre style="background:#1A2E45;color:#FFB347;padding:1rem;border-radius:8px;font-size:0.8rem;overflow-x:auto;white-space:pre;">{json.dumps(getattr(import_result, "detected_columns", {}), indent=2)}</pre>', unsafe_allow_html=True)

    for w in getattr(import_result, "warnings", []) or []:
        st.warning(w)

    st.subheader("Normalized Preview (engine-ready)")
    st.dataframe(import_result.df.head(10))

    safe_label = label.lower().replace(" ", "_")
    st.download_button(
        label=f"Download Normalized CSV ({label})",
        data=import_result.df.to_csv(index=False).encode("utf-8"),
        file_name=f"normalized_{safe_label}.csv",
        mime="text/csv",
        key=f"download_norm_{safe_label}",
    )

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

    st.subheader("Decision")
    status = decision.get("status", "DECISION_OK")
    tier = decision.get("confidence_tier", "n/a")
    action = decision.get("action", "HOLD")

    if status == "DECISION_BLOCKED":
        st.error("Decision Outcome: BLOCK (Safety Outcome)")
        st.write("**Blocking reason:**", decision.get("reason", ""))
    elif status == "DECISION_WARN":
        st.warning("Decision issued with caution.")
    else:
        st.success("Decision issued (non-executing).")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Decision Outcome", "BLOCK" if status == "DECISION_BLOCKED" else action)
    with c2:
        st.metric("Confidence Tier", tier)
    with c3:
        st.metric("Budget change", decision.get("recommended_change_pct_range", "n/a"))

    st.write("**Decision Mode:**", decision.get("decision_mode", "n/a"))
    st.write("**Primary Constraint:**", decision.get("primary_constraint", "n/a"))
    st.write("**Reason (factual):**", decision.get("reason", ""))
    st.write("**User Explanation:**", decision.get("user_explanation", ""))

    st.subheader("Structured explanation (audit-friendly)")
    se = decision.get("structured_explanation", {}) or {}
    if se:
        st.write(f"**Decision Outcome:** {se.get('Decision Outcome', '')}")
        st.write(f"**Primary Constraint:** {se.get('Primary Constraint', '')}")
        st.write("**Supporting Factors:**")
        for f in se.get("Supporting Factors", []) or []:
            st.write(f"- {f}")
        st.write(f"**Confidence Level:** {se.get('Confidence Level', '')}")
        st.write(f"**Operator Consideration:** {se.get('Operator Consideration', '')}")
    else:
        st.info("Structured explanation unavailable.")

    st.caption(f"Next review: {decision.get('next_review_window', 'n/a')}")
    show_override_log_prompt(
        decision_id=ih or str(result.get('random_seed', '')),
        recommendation="BLOCK" if status == "DECISION_BLOCKED" else action,
        confidence_score=float(result.get("decision_confidence", 0.0)),
        campaign_name=label,
        risk_profile=profile_name,
        user_email=st.session_state.get("user_email", ""),
    )

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

    st.subheader("Summary")
    if status == "DECISION_BLOCKED":
        st.code(f"Outcome: BLOCK • Primary constraint: {decision.get('primary_constraint')} • Reason: {decision.get('reason')}")
    else:
        st.code(recommendation_summary(result, decision, profile.name))

    if st.button(f"Generate Report File ({label})", key=f"report_{label}"):
        result["spend_total"] = float(import_result.df["spend"].sum()) if "spend" in import_result.df.columns else float(result.get("spend_total", 0.0))
        result["simulations"] = int(simulations)
        result["signal_reliability"] = float(signal_reliability)
        result["scale_pct"] = float(scale_pct)
        if not result.get("random_seed"):
            result["random_seed"] = stable_seed_from_df(import_result.df)

        report_path = write_markdown_report(result, decision, profile.name, platform_label=label)
        st.success(f"Report saved: {report_path}")

        try:
            with open(report_path, "rb") as f:
                st.download_button(label=f"Download Report (Markdown) — {label}", data=f.read(), file_name=report_path.split("/")[-1], mime="text/markdown", key=f"download_md_{label}")
        except Exception as e:
            st.warning(f"Could not load report file for download: {e}")

        snapshot = {
            "snapshot_type": "channel_decision", "platform": label, "logged_at_utc": utc_now_iso(),
            "engine_version": result.get("engine_version"), "ruleset_version": result.get("ruleset_version"),
            "random_seed": int(result.get("random_seed") or 0),
            "input_hash": decision.get("input_hash") or result.get("input_hash"),
            "simulations": int(result.get("simulations") or 0),
            "signal_reliability": float(result.get("signal_reliability") or 0.0),
            "scale_pct": float(result.get("scale_pct") or 0.0),
            "days_of_data": int(result.get("days_of_data") or 0),
            "date_min": result.get("date_min"), "date_max": result.get("date_max"),
            "validation": result.get("validation", {}), "decision": decision,
        }
        snap_copy = dict(snapshot)
        snap_copy.pop("snapshot_hash", None)
        snapshot["snapshot_hash"] = hashlib.sha256(json.dumps(snap_copy, sort_keys=True).encode("utf-8")).hexdigest()

        st.download_button(
            label=f"Download Audit Snapshot (JSON) — {label}",
            data=json.dumps(snapshot, indent=2).encode("utf-8"),
            file_name=f"mdu_snapshot_{safe_label}.json",
            mime="application/json",
            key=f"download_snapshot_{label}",
        )

        try:
            pdf_bytes = build_audit_pdf_bytes(label, profile.name, result, decision, snapshot)
            st.download_button(label=f"Download Report (PDF) — {label}", data=pdf_bytes, file_name=f"mdu_report_{safe_label}.pdf", mime="application/pdf", key=f"download_pdf_{label}")
        except Exception as e:
            st.warning(f"PDF generation failed: {e}")

        log_decision({
            "type": "channel", "platform": label, "status": decision.get("status"), "action": decision.get("action"),
            "decision_mode": decision.get("decision_mode"), "primary_constraint": decision.get("primary_constraint"),
            "confidence_tier": decision.get("confidence_tier"), "confidence": float(result.get("decision_confidence", 0.0)),
            "downside_risk": float(result.get("downside_risk", 1.0)), "spend_total": float(result.get("spend_total", 0.0)),
            "days_of_data": int(result.get("days_of_data", 0)), "date_min": result.get("date_min"), "date_max": result.get("date_max"),
            "engine_version": result.get("engine_version"), "ruleset_version": result.get("ruleset_version"),
            "random_seed": int(result.get("random_seed") or 0),
            "recommended_change_pct_range": decision.get("recommended_change_pct_range"),
            "next_review_window": decision.get("next_review_window"),
            "validation_status": (result.get("validation", {}) or {}).get("status"),
            "block_reason": (result.get("validation", {}) or {}).get("block_reason"),
            "input_hash": decision.get("input_hash") or result.get("input_hash"),
            "logged_at_utc": utc_now_iso(),
        })

# ── Portfolio decision ────────────────────────────────────
st.divider()
st.header("Portfolio Decision (Cross-channel)")
st.subheader("Portfolio Controls")
pc1, pc2, pc3 = st.columns(3)
with pc1:
    min_portfolio_conf = st.slider("Min portfolio confidence", 0.0, 1.0, 0.60, 0.05)
with pc2:
    min_signal_separation = st.number_input("Min signal separation (avg net value delta)", min_value=0.0, value=500.0, step=50.0)
with pc3:
    max_allowed_downside_risk = st.slider("Max allowed downside risk (to reallocate into)", 0.0, 1.0, 0.55, 0.05)

if len(channels_for_portfolio) >= 2:
    try:
        portfolio = recommend_portfolio_action(
            channels_for_portfolio, risk_profile=profile_name,
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
        for x in portfolio.rationale_blocks.get("what_happened", []):
            st.write(f"- {x}")
        for x in portfolio.rationale_blocks.get("what_could_go_wrong", []):
            st.write(f"- {x}")
        for x in portfolio.rationale_blocks.get("what_to_do_next", []):
            st.write(f"- {x}")
        if st.button("Log Portfolio Decision", key="log_portfolio"):
            log_decision({
                "type": "portfolio", "platform": "Portfolio", "action": portfolio.portfolio_action,
                "portfolio_confidence": float(portfolio.portfolio_confidence), "reallocation_enabled": bool(rec.enabled),
                "from_platform": rec.from_platform if rec.enabled else None,
                "to_platform": rec.to_platform if rec.enabled else None,
                "amount": float(rec.amount) if rec.enabled else 0.0,
                "engine_version": ENGINE_VERSION, "ruleset_version": RULESET_VERSION, "logged_at_utc": utc_now_iso(),
            })
            st.success("Portfolio decision logged to history.")
    except Exception as e:
        st.error(f"Portfolio decision error: {e}")
else:
    st.info("Upload both Meta and Google files (and pass validation) to get a portfolio outcome.")

# ── Snapshot Replay ───────────────────────────────────────
st.divider()
st.header("Replay a Snapshot (Audit)")
st.caption("Upload an audit snapshot JSON previously generated by MDU Engine.")
snapshot_file = st.file_uploader("Upload Snapshot JSON", type=["json"], key="snapshot_replay")
if snapshot_file:
    try:
        snapshot = json.loads(snapshot_file.getvalue().decode("utf-8", errors="replace"))
        ok, errors, warnings = validate_snapshot(snapshot)
        st.subheader("Snapshot Preview")
        st.markdown(f'<pre style="background:#1A2E45;color:#FFB347;padding:1rem;border-radius:8px;font-size:0.8rem;overflow-x:auto;white-space:pre;">{json.dumps(snapshot, indent=2)}</pre>', unsafe_allow_html=True)
        st.subheader("Replay Result")
        if ok:
            st.success("REPLAY PASS ✅ — Snapshot is structurally valid and audit-ready.")
        else:
            st.error("REPLAY FAIL ❌ — Snapshot failed validation checks.")
        for w in warnings:
            st.write(f"- {w}")
        for e in errors:
            st.write(f"- {e}")
        if st.button("Log Replay Result", key="log_replay"):
            log_decision({
                "type": "replay", "platform": snapshot.get("platform", "unknown"), "replay_pass": bool(ok),
                "snapshot_type": snapshot.get("snapshot_type"), "engine_version": snapshot.get("engine_version"),
                "ruleset_version": snapshot.get("ruleset_version"), "random_seed": int(snapshot.get("random_seed") or 0),
                "simulations": int(snapshot.get("simulations") or 0),
                "signal_reliability": float(snapshot.get("signal_reliability") or 0.0),
                "scale_pct": float(snapshot.get("scale_pct") or 0.0), "days_of_data": int(snapshot.get("days_of_data") or 0),
                "date_min": snapshot.get("date_min"), "date_max": snapshot.get("date_max"), "logged_at_utc": utc_now_iso(),
            })
            st.success("Replay result logged to history.")
    except Exception as e:
        st.error(f"Could not read/parse snapshot JSON: {e}")

# ── Decision History Dashboard ────────────────────────────
st.divider()
st.header("Decision History")
st.caption("All decision runs are automatically logged. History persists across sessions.")

history = read_history(limit=100)

if not history:
    st.info("No decision history yet. Run a decision above to start building your audit trail.")
else:
    dfh = pd.DataFrame(history).fillna("")

    # ── Summary metrics ───────────────────────────────────
    total_runs = len(dfh)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Runs", total_runs)
    with col2:
        scale_count = len(dfh[dfh.get("action", pd.Series(dtype=str)).eq("SCALE")]) if "action" in dfh.columns else 0
        st.metric("SCALE outcomes", scale_count)
    with col3:
        hold_count = len(dfh[dfh.get("action", pd.Series(dtype=str)).eq("HOLD")]) if "action" in dfh.columns else 0
        st.metric("HOLD outcomes", hold_count)
    with col4:
        block_count = len(dfh[dfh.get("status", pd.Series(dtype=str)).eq("DECISION_BLOCKED")]) if "status" in dfh.columns else 0
        st.metric("BLOCK outcomes", block_count)

    # ── Outcome distribution chart ─────────────────────────
    if "action" in dfh.columns and "status" in dfh.columns:
        st.subheader("Outcome Distribution")

        def get_final_outcome(row):
            if row.get("status") == "DECISION_BLOCKED":
                return "BLOCK"
            return row.get("action", "UNKNOWN")

        dfh["final_outcome"] = dfh.apply(get_final_outcome, axis=1)
        outcome_counts = dfh["final_outcome"].value_counts().reset_index()
        outcome_counts.columns = ["Outcome", "Count"]

        fig1 = px.bar(
            outcome_counts, x="Outcome", y="Count",
            color="Outcome",
            color_discrete_map={"SCALE": "#00C896", "HOLD": "#FFB347", "REDUCE": "#1E90FF", "BLOCK": "#FF4444"},
            template="plotly_dark",
        )
        fig1.update_layout(
            paper_bgcolor="#0D1B2A",
            plot_bgcolor="#0D1B2A",
            font_color="#E8F0FE",
            showlegend=False,
        )
        st.plotly_chart(fig1, use_container_width=True)

    # ── Confidence trend chart ─────────────────────────────
    if "confidence" in dfh.columns and "logged_at_utc" in dfh.columns:
        st.subheader("Decision Confidence Over Time")
        df_trend = dfh[["logged_at_utc", "confidence", "platform"]].copy()
        df_trend = df_trend[df_trend["confidence"] != ""]
        df_trend["confidence"] = pd.to_numeric(df_trend["confidence"], errors="coerce")
        df_trend["logged_at_utc"] = pd.to_datetime(df_trend["logged_at_utc"], errors="coerce")
        df_trend = df_trend.dropna(subset=["logged_at_utc", "confidence"]).sort_values("logged_at_utc")
        if not df_trend.empty:
            import plotly.express as px
            fig2 = px.line(
                df_trend, x="logged_at_utc", y="confidence",
                template="plotly_dark",
                labels={"logged_at_utc": "Date", "confidence": "Confidence Score"},
            )
            fig2.update_layout(
                paper_bgcolor="#0D1B2A",
                plot_bgcolor="#0D1B2A",
                font_color="#E8F0FE",
            )
            fig2.update_traces(line_color="#1E90FF")
            st.plotly_chart(fig2, use_container_width=True)

    # ── Platform filter ────────────────────────────────────
    st.subheader("History Log")

    if "platform" in dfh.columns:
        platforms = ["All"] + sorted(dfh["platform"].dropna().unique().tolist())
        selected_platform = st.selectbox("Filter by platform", platforms, key="history_platform_filter")
        if selected_platform != "All":
            dfh = dfh[dfh["platform"] == selected_platform]

    # ── Clean table display ────────────────────────────────
    preferred = [
        "logged_at_utc", "type", "platform", "final_outcome",
        "confidence", "downside_risk", "days_of_data",
        "date_min", "date_max", "spend_total",
        "engine_version", "ruleset_version", "input_hash"
    ]
    display_cols = [c for c in preferred if c in dfh.columns]
    remaining = [c for c in dfh.columns if c not in display_cols]
    display_cols = display_cols + remaining

    st.dataframe(dfh[display_cols].head(50), use_container_width=True)

    # ── Downloads ──────────────────────────────────────────
    show_override_dashboard()
    st.divider()
    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        st.download_button(
            label="Download History (CSV)",
            data=dfh.to_csv(index=False).encode("utf-8"),
            file_name="mdu_decision_history.csv",
            mime="text/csv",
            key="download_history_csv",
        )

    with dl_col2:
        from mdu_engine.history import history_to_json_bytes
        st.download_button(
            label="Download History (JSON)",
            data=history_to_json_bytes(limit=500),
            file_name="mdu_decision_history.json",
            mime="application/json",
            key="download_history_json",
        )

    # ── Latest decision audit snapshot ────────────────────
    latest = get_latest_decision()
    if latest:
        with st.expander("Latest Decision — Audit Snapshot", expanded=False):
            st.markdown(f'<pre style="background:#1A2E45;color:#FFB347;padding:1rem;border-radius:8px;font-size:0.8rem;overflow-x:auto;white-space:pre;">{json.dumps(latest, indent=2)}</pre>', unsafe_allow_html=True)


# ── Feedback Section ──────────────────────────────────────
st.divider()
st.header("Share Your Feedback (Optional)")
st.caption("Your feedback helps improve MDU Engine. Takes 30 seconds. Completely optional. No account required.")

_feedback_platform = _feedback_outcome = _feedback_tier = _feedback_engine = _feedback_ruleset = _feedback_hash = ""
_feedback_days = 0
_feedback_spend = 0.0

if channel_outputs:
    _last_label = list(channel_outputs.keys())[-1]
    _, _, _last_import, _last_result, _last_decision = channel_outputs[_last_label]
    _feedback_platform = _last_label
    _feedback_outcome = "BLOCK" if _last_decision.get("status") == "DECISION_BLOCKED" else _last_decision.get("action", "")
    _feedback_days = int(_last_result.get("days_of_data") or 0)
    _feedback_spend = float(_last_result.get("spend_total") or 0.0)
    _feedback_tier = _last_decision.get("confidence_tier", "")
    _feedback_engine = _last_result.get("engine_version", "")
    _feedback_ruleset = _last_result.get("ruleset_version", "")
    _feedback_hash = _last_decision.get("input_hash") or _last_result.get("input_hash") or ""

with st.form("feedback_form", clear_on_submit=True):
    st.markdown("**Tell us about your experience**")
    fb_col1, fb_col2 = st.columns(2)
    with fb_col1:
        fb_name = st.text_input("Your name", placeholder="e.g. Priya Sharma")
        fb_role = st.text_input("Your role / title", placeholder="e.g. Performance Marketing Lead")
    with fb_col2:
        fb_company = st.text_input("Company or context", placeholder="e.g. Agency name, startup, freelance")
        fb_useful = st.radio("Was this evaluation useful?", options=["Yes", "Partially", "No"], horizontal=True)
    fb_use_case = st.text_area("What did you use this evaluation for?", placeholder="e.g. Deciding whether to scale our Meta campaign after a CPL spike.", max_chars=500, height=80)
    fb_submitted = st.form_submit_button("Submit Feedback", use_container_width=True, type="primary")

    if fb_submitted:
        if not fb_name.strip() or not fb_role.strip():
            st.warning("Please enter at least your name and role.", icon="⚠️")
        else:
            record = {
                "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
                "name": fb_name.strip(), "role": fb_role.strip(), "company": fb_company.strip(),
                "use_case": fb_use_case.strip(), "was_useful": fb_useful,
                "decision_outcome": _feedback_outcome, "platform": _feedback_platform,
                "days_of_data": _feedback_days, "spend_total": round(_feedback_spend, 2),
                "confidence_tier": _feedback_tier, "engine_version": _feedback_engine,
                "ruleset_version": _feedback_ruleset, "input_hash": _feedback_hash,
            }
            ok = write_feedback(record)
            if ok:
                st.success("Thank you. Your feedback has been recorded.", icon="✅")
            else:
                st.warning("Feedback could not be saved. Please try again.", icon="⚠️")

with st.expander("Feedback History (Admin)", expanded=False):
    fb_history = read_feedback(limit=50)
    if not fb_history:
        st.info("No feedback submitted yet.")
    else:
        st.caption(f"{len(fb_history)} feedback record(s) on file.")
        fb_csv = feedback_to_csv_bytes()
        if fb_csv:
            st.download_button("Download All Feedback (CSV)", data=fb_csv, file_name="mdu_engine_feedback.csv", mime="text/csv", key="download_feedback_csv")
        _dfb = pd.DataFrame(fb_history)
        _preferred_cols = ["submitted_at_utc", "name", "role", "company", "was_useful", "decision_outcome", "platform", "use_case", "days_of_data", "spend_total"]
        _cols = [c for c in _preferred_cols if c in _dfb.columns] + [c for c in _dfb.columns if c not in _preferred_cols]
        st.dataframe(_dfb[_cols], use_container_width=True)

st.divider()
st.caption("© 2026 Satish Saka · MDU Engine · MIT License · Public Decision Engine")
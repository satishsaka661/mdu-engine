"""
override_ui_section.py — MDU Engine Override Logging UI
Paste this into app.py after the main decision output section.
"""

import streamlit as st
from datetime import datetime
from mdu_engine.override_log import (
    create_override_log,
    record_outcome,
    get_pending_followups,
    get_override_summary,
)


# ── Section 1: Log a new decision after recommendation is shown ───────────────

def show_override_log_prompt(
    decision_id: str,
    recommendation: str,
    confidence_score: float,
    campaign_name: str = "",
    risk_profile: str = "",
):
    """
    Call this immediately after displaying the SCALE/HOLD/REDUCE/BLOCK output.
    Creates the pending log entry and shows a confirmation to the user.
    """
    st.markdown("---")
    st.markdown("#### 📋 Decision Logged")

    with st.container():
        st.markdown(
            f"""
            <div style="background:#0D1B2A; border:1px solid #1E90FF;
                        border-radius:8px; padding:16px; margin-bottom:12px;">
                <p style="color:#ccc; margin:0; font-size:14px;">
                    This recommendation has been logged. You'll be prompted in
                    <strong style="color:#1E90FF;">48 hours</strong> to record
                    what you did and what happened next.
                </p>
                <p style="color:#666; margin:8px 0 0 0; font-size:12px;">
                    Your override data helps identify where the system's
                    calibration needs improvement.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Store log_id in session state so the followup form can reference it
    if "pending_log_ids" not in st.session_state:
        st.session_state.pending_log_ids = []

    log_id = create_override_log(
        decision_id=decision_id,
        recommendation=recommendation,
        confidence_score=confidence_score,
        campaign_name=campaign_name,
        risk_profile=risk_profile,
    )

    # Avoid duplicate logging on reruns
    if log_id not in st.session_state.pending_log_ids:
        st.session_state.pending_log_ids.append(log_id)

    return log_id


# ── Section 2: 48-hour follow-up form ─────────────────────────────────────────

def show_followup_forms():
    """
    Shows follow-up forms for any pending decisions that are now due (48hrs elapsed).
    Call this near the top of app.py so it surfaces on every session.
    """
    pending = get_pending_followups()
    if not pending:
        return

    st.markdown("---")
    st.markdown("## ⏱ 48-Hour Follow-Up")
    st.markdown(
        "These decisions are ready for your outcome report. "
        "This data helps calibrate MDU Engine's recommendations over time."
    )

    for log in pending:
        rec = log["recommendation"]
        created = log["created_at"][:10]
        campaign = log.get("campaign_name", "Unknown campaign")
        log_id = log["log_id"]

        color_map = {
            "SCALE": "#00C851",
            "HOLD": "#FF8C00",
            "REDUCE": "#FF4444",
            "BLOCK": "#CC0000",
        }
        color = color_map.get(rec, "#1E90FF")

        with st.expander(
            f"📌 {rec} — {campaign} — logged {created}", expanded=True
        ):
            st.markdown(
                f"""
                <div style="background:#0D1B2A; border-left:4px solid {color};
                            padding:12px; border-radius:4px; margin-bottom:16px;">
                    <span style="color:{color}; font-weight:700; font-size:18px;">
                        {rec}
                    </span>
                    <span style="color:#aaa; font-size:13px; margin-left:12px;">
                        Confidence: {log.get('confidence_score', 0):.0%}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            was_followed = st.radio(
                "Did you follow this recommendation?",
                ["Yes, I followed it", "No, I overrode it"],
                key=f"followed_{log_id}",
                horizontal=True,
            )

            override_reason = None
            if "No" in was_followed:
                override_reason = st.text_area(
                    "What did you do instead, and why?",
                    placeholder="e.g. Client insisted on scaling despite HOLD. Budget pressure from Q4 targets.",
                    key=f"reason_{log_id}",
                    height=80,
                )

            outcome_direction = st.radio(
                "What happened to campaign performance in the 48 hours after?",
                ["Improved", "Unchanged", "Worsened"],
                key=f"direction_{log_id}",
                horizontal=True,
            )

            outcome_description = st.text_area(
                "Brief description of what happened (optional)",
                placeholder="e.g. CPA recovered to baseline without intervention. ROAS dropped further after scaling.",
                key=f"outcome_{log_id}",
                height=80,
            )

            if st.button("Submit outcome", key=f"submit_{log_id}"):
                success = record_outcome(
                    log_id=log_id,
                    was_followed="Yes" in was_followed,
                    override_reason=override_reason,
                    outcome_description=outcome_description,
                    outcome_direction=outcome_direction.lower(),
                )
                if success:
                    st.success("Outcome recorded. Thank you — this improves calibration.")
                    st.rerun()
                else:
                    st.error("Something went wrong saving the outcome. Please try again.")


# ── Section 3: Override Intelligence Dashboard ─────────────────────────────────

def show_override_dashboard():
    """
    Shows aggregate override stats.
    Add this to your Decision History Dashboard section.
    """
    summary = get_override_summary()

    if summary.get("completed", 0) == 0:
        st.info(
            "Override intelligence builds as decisions are logged and outcomes recorded. "
            "No completed follow-ups yet."
        )
        return

    st.markdown("### 🧠 Override Intelligence")
    st.markdown(
        "Tracks where recommendations were followed vs overridden "
        "and whether outcomes validate the system's calibration."
    )

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Decisions Logged", summary["total"])
    col2.metric("Follow Rate", f"{summary['follow_rate']}%")
    col3.metric("Override Rate", f"{summary['override_rate']}%")
    col4.metric("Pending Follow-ups", summary.get("pending", 0))

    st.markdown("---")

    # Override rate by action class
    st.markdown("#### Override Rate by Action Class")
    override_data = summary.get("override_by_class", {})

    cols = st.columns(4)
    color_map = {
        "SCALE": "#00C851",
        "HOLD": "#FF8C00",
        "REDUCE": "#FF4444",
        "BLOCK": "#CC0000",
    }
    for i, (action, stats) in enumerate(override_data.items()):
        with cols[i]:
            st.markdown(
                f"""
                <div style="background:#0D1B2A; border:1px solid {color_map[action]};
                            border-radius:8px; padding:12px; text-align:center;">
                    <div style="color:{color_map[action]}; font-weight:700;
                                font-size:16px;">{action}</div>
                    <div style="color:#fff; font-size:22px; font-weight:700;">
                        {stats['override_rate']}%
                    </div>
                    <div style="color:#888; font-size:11px;">
                        {stats['overridden']}/{stats['total']} overridden
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Outcome correlation
    st.markdown("#### Outcome Correlation")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"""
            <div style="background:#0D1B2A; border:1px solid #00C851;
                        border-radius:8px; padding:16px; text-align:center;">
                <div style="color:#aaa; font-size:12px;">When recommendation followed</div>
                <div style="color:#00C851; font-size:28px; font-weight:700;">
                    {summary['outcome_when_followed_improved']}%
                </div>
                <div style="color:#888; font-size:12px;">improved</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"""
            <div style="background:#0D1B2A; border:1px solid #FF4444;
                        border-radius:8px; padding:16px; text-align:center;">
                <div style="color:#aaa; font-size:12px;">When recommendation overridden</div>
                <div style="color:#FF4444; font-size:28px; font-weight:700;">
                    {summary['outcome_when_overridden_improved']}%
                </div>
                <div style="color:#888; font-size:12px;">improved</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Miscalibration flags
    flags = summary.get("miscalibration_flags", [])
    if flags:
        st.markdown("---")
        st.markdown("#### ⚠️ Miscalibration Flags")
        st.warning(
            f"The following action classes have >50% override rate across 3+ decisions, "
            f"suggesting the system may be miscalibrated for these outputs: "
            f"**{', '.join(flags)}**"
        )
        st.markdown(
            "_These flags surface where the system's thresholds may need review._"
        )
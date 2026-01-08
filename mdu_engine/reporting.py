from __future__ import annotations

from datetime import datetime
from pathlib import Path


def confidence_band(confidence: float) -> str:
    if confidence >= 0.70:
        return "HIGH"
    if confidence >= 0.45:
        return "MEDIUM"
    return "LOW"


def risk_band(downside_risk: float) -> str:
    if downside_risk >= 0.55:
        return "HIGH"
    if downside_risk >= 0.40:
        return "MEDIUM"
    return "LOW"


def recommendation_summary(result: dict, decision: dict, profile_name: str) -> str:
    conf = float(result.get("decision_confidence", 0.0))
    risk = float(result.get("downside_risk", 0.0))

    conf_b = confidence_band(conf)
    risk_b = risk_band(risk)

    lines = []
    lines.append("=== Recommendation Summary ===")
    lines.append(f"risk_profile: {profile_name}")
    lines.append(f"action: {decision.get('action')}")
    lines.append(f"confidence: {conf:.3f} ({conf_b})")
    lines.append(f"downside_risk: {risk:.3f} ({risk_b})")
    lines.append(f"why: {decision.get('user_explanation')}")
    return "\n".join(lines)


def write_markdown_report(
    result: dict,
    decision: dict,
    profile_name: str,
    platform_label: str = "Channel",
) -> str:
    """
    Creates a markdown report under ./reports and returns the file path.
    """

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe_platform = platform_label.lower().replace(" ", "_")
    filename = f"decision_report_{safe_platform}_{today}.md"
    path = reports_dir / filename

    # Safe getters
    conf = float(result.get("decision_confidence", 0.0))
    risk = float(result.get("downside_risk", 0.0))
    # --- Audit / versioning fields (industry standard) ---
    engine_version = result.get("engine_version", "n/a")
    ruleset_version = result.get("ruleset_version", "n/a")
    random_seed = result.get("random_seed") or "n/a"

    validation = result.get("validation", {}) or {}
    validation_status = validation.get("status", "n/a")
    block_reason = validation.get("block_reason", "")

    content: list[str] = []
    content.append("# MDU Engine — Decision Report\n")
    content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    content.append(f"**Platform:** {platform_label}\n")
    content.append(f"**Risk Profile:** {profile_name}\n")

    # ✅ ADD THESE LINES
    content.append(f"**Engine Version:** {engine_version}\n")
    content.append(f"**Ruleset Version:** {ruleset_version}\n")
    content.append(f"**Random Seed:** {random_seed}\n")
    content.append(f"**Validation Status:** {validation_status}\n")
    if block_reason:
     content.append(f"**Block Reason:** {block_reason}\n")
    content.append("\n---\n")

    # Summary block
    content.append("## Recommendation Summary\n")
    content.append("```")
    content.append(recommendation_summary(result, decision, profile_name))
    content.append("```")
    content.append("\n---\n")

    # Decision details
    content.append("## Decision Details\n")
    content.append(f"- action: {decision.get('action')}\n")
    content.append(f"- reason: {decision.get('reason')}\n")
    content.append(f"- user_explanation: {decision.get('user_explanation')}\n")
    content.append("\n---\n")

    # Explainability (if present)
    exp = decision.get("explainability")
    if exp:
        content.append("## Why this decision?\n")

        content.append("### What happened\n")
        for x in exp.get("what_happened", []):
            content.append(f"- {x}\n")

        content.append("\n### What could go wrong\n")
        for x in exp.get("what_could_go_wrong", []):
            content.append(f"- {x}\n")

        content.append("\n### What to do next\n")
        for x in exp.get("what_to_do_next", []):
            content.append(f"- {x}\n")

        content.append("\n---\n")

    # ✅ Data Window (coverage)
    content.append("## Data Window\n")
    date_min = result.get("date_min")
    date_max = result.get("date_max")
    days_of_data = result.get("days_of_data")

    if date_min and date_max:
        content.append(f"- date_range: {date_min} → {date_max}\n")
    else:
        content.append("- date_range: n/a\n")

    if days_of_data is not None:
        content.append(f"- days_of_data: {days_of_data}\n")
    else:
        content.append("- days_of_data: n/a\n")

    content.append("\n---\n")

    # Raw metrics
    content.append("## Raw Metrics\n")
    content.append(f"- decision_confidence: {conf}\n")
    content.append(f"- downside_risk: {risk}\n")
    content.append(f"- avg_net_value: {result.get('avg_net_value', 'n/a')}\n")
    content.append(f"- spend_total: {result.get('spend_total', 'n/a')}\n")
    content.append(f"- simulations: {result.get('simulations', 'n/a')}\n")
    content.append(f"- signal_reliability: {result.get('signal_reliability', 'n/a')}\n")
    content.append(f"- scale_pct: {result.get('scale_pct', 'n/a')}\n")

    text = "\n".join(content)
    path.write_text(text, encoding="utf-8")

    # latest.md pointer
    latest_path = reports_dir / "latest.md"
    latest_path.write_text(text, encoding="utf-8")

    return str(path)
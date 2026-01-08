from mdu_engine.decision_confidence import load_data, compute_decision_confidence
from mdu_engine.decision_rules import RISK_PROFILES, decide_action
from mdu_engine.reporting import recommendation_summary, write_markdown_report


# 1) Load data
df = load_data("data/sample_daily.csv")

# 2) Compute confidence metrics
result = compute_decision_confidence(
    df,
    signal_reliability=0.6,
    scale_pct=0.10,
    simulations=5000,
)

# 3) Choose a risk profile (start with balanced)
profile = RISK_PROFILES["balanced"]

# 4) Make a decision
decision = decide_action(
    decision_confidence=result["decision_confidence"],
    downside_risk=result["downside_risk"],
    profile=profile,
)
print("\n" + recommendation_summary(result, decision, profile.name))
report_path = write_markdown_report(result, decision, profile.name)
print(f"\n✅ Report saved to: {report_path}")


print("\n=== Decision Rule Output ===")
print(f"risk_profile: {profile.name}")
print(f"action: {decision['action']}")
print(f"reason: {decision['reason']}")

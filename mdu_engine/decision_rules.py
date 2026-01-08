from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RiskProfile:
    name: str
    scale_threshold: float
    maintain_threshold: float
    max_downside_risk: float
    reduce_downside_risk: float  # if risk is extreme -> REDUCE


RISK_PROFILES: Dict[str, RiskProfile] = {
    "conservative": RiskProfile(
        name="conservative",
        scale_threshold=0.70,
        maintain_threshold=0.55,
        max_downside_risk=0.30,
        reduce_downside_risk=0.45,
    ),
    "balanced": RiskProfile(
        name="balanced",
        scale_threshold=0.60,
        maintain_threshold=0.45,
        max_downside_risk=0.40,
        reduce_downside_risk=0.55,
    ),
    "growth": RiskProfile(
        name="growth",
        scale_threshold=0.50,
        maintain_threshold=0.40,
        max_downside_risk=0.50,
        reduce_downside_risk=0.65,
    ),
}


def decide_action(decision_confidence: float, downside_risk: float, profile: RiskProfile, days_of_data: int | None = None) -> dict:
    """
    Actions:
    - SCALE: evidence is strong and risk is acceptable
    - MAINTAIN: keep steady, gather more data
    - HOLD: pause scaling until confidence improves
    - REDUCE: risk is high enough that we should cut spend
    """
        # ✅ Data quality gate: don't take strong actions with too little data
    if days_of_data is not None and days_of_data < 7:
        return {
            "action": "HOLD",
            "reason": f"Only {days_of_data} day(s) of data. Need at least 7 days for a reliable decision.",
            "user_explanation": "Hold spend steady. Upload 7–30 days of data to get a stable recommendation.",
            "explainability": {
                "what_happened": [f"Only {days_of_data} day(s) of data were available after cleaning."],
                "what_could_go_wrong": ["With too little data, confidence and risk estimates can be misleading."],
                "what_to_do_next": ["Export a report segmented by Day and upload at least 7–30 days."],
            },
        }

    # Rule 0: If downside risk is extreme, reduce immediately
    if downside_risk >= profile.reduce_downside_risk:
        return {
            "action": "REDUCE",
            "reason": (
                f"Downside risk is extreme ({downside_risk:.3f} ≥ {profile.reduce_downside_risk:.2f}). "
                f"Reduce spend and investigate."
            ),
            "user_explanation": (
                "Risk is too high right now. Reduce budget and diagnose issues "
                "(tracking, creatives, audience, landing page) before scaling again."
            ),
            "explainability": {
                "what_happened": [
                    f"Downside risk spiked to {downside_risk:.3f}, above the REDUCE threshold for {profile.name}.",
                    "The system sees a high chance of negative net outcomes at current allocation.",
                ],
                "what_could_go_wrong": [
                    "Continuing current spend may amplify losses while the signal is unstable.",
                    "If tracking/conversion quality is broken, decisions will be misleading.",
                ],
                "what_to_do_next": [
                    "Reduce spend immediately to control downside exposure.",
                    "Validate tracking and conversion integrity; fix data quality issues.",
                    "Re-run after 7 more days of stable data before scaling again.",
                ],
            },
        }

    # Rule 1: If downside risk is above safe limit, don't scale
    if downside_risk > profile.max_downside_risk:
        return {
            "action": "HOLD",
            "reason": (
                f"Downside risk ({downside_risk:.3f}) is above the profile limit "
                f"({profile.max_downside_risk:.2f}) for '{profile.name}'."
            ),
            "user_explanation": (
                "Don’t scale yet. There’s a meaningful chance performance will worsen if you increase spend."
            ),
            "explainability": {
                "what_happened": [
                    f"Downside risk is {downside_risk:.3f}, above the safe scaling band for {profile.name}.",
                    "Risk is currently too elevated to justify increasing allocation.",
                ],
                "what_could_go_wrong": [
                    "Scaling now increases the chance of wasted spend during unstable performance.",
                    "A short-term uptick could reverse quickly, creating avoidable losses.",
                ],
                "what_to_do_next": [
                    "Hold spend steady until risk returns to the acceptable range.",
                    "Collect more daily data to confirm stability before scaling.",
                ],
            },
        }

    # Rule 2: Scale only if confidence is high enough
    if decision_confidence >= profile.scale_threshold:
        return {
            "action": "SCALE",
            "reason": (
                f"Confidence ({decision_confidence:.3f}) meets scale threshold "
                f"({profile.scale_threshold:.2f}) for '{profile.name}'."
            ),
            "user_explanation": (
                "You have strong evidence. Scale gradually (e.g., +10%) and monitor results."
            ),
            "explainability": {
                "what_happened": [
                    f"Decision confidence is {decision_confidence:.3f}, above the SCALE threshold for {profile.name}.",
                    "The signal appears stable enough to allocate additional budget with controlled risk.",
                ],
                "what_could_go_wrong": [
                    "Scaling too fast can change auction dynamics and reduce efficiency.",
                    "If recent performance is driven by a temporary spike, results may regress.",
                ],
                "what_to_do_next": [
                    "Scale in controlled steps (e.g., +10%).",
                    "Re-run after each step and stop if downside risk rises materially.",
                ],
            },
        }

    # Rule 3: Maintain if confidence is moderate
    if decision_confidence >= profile.maintain_threshold:
        return {
            "action": "MAINTAIN",
            "reason": (
                f"Confidence ({decision_confidence:.3f}) is moderate (>= {profile.maintain_threshold:.2f})."
            ),
            "user_explanation": (
                "Keep spend steady and collect more data before making bigger budget decisions."
            ),
            "explainability": {
                "what_happened": [
                    f"Decision confidence is {decision_confidence:.3f} (moderate) for {profile.name}.",
                    "The signal is forming but not strong enough to scale safely yet.",
                ],
                "what_could_go_wrong": [
                    "Scaling early can lock in spend before the signal becomes reliable.",
                    "Reducing too soon could cut off learning and delay stabilization.",
                ],
                "what_to_do_next": [
                    "Maintain spend and wait for stronger confirmation in the data.",
                    "Re-run after 7–14 days or after meaningful new data arrives.",
                ],
            },
        }

    # Rule 4: Otherwise hold
    return {
        "action": "HOLD",
        "reason": f"Confidence ({decision_confidence:.3f}) is too low to act safely.",
        "user_explanation": (
            "Not enough reliable evidence to scale. Keep steady and improve data quality or wait for more days."
        ),
        "explainability": {
            "what_happened": [
                f"Decision confidence is {decision_confidence:.3f}, below the minimum safe threshold for action.",
                "The system cannot reliably estimate outcomes if allocation is changed.",
            ],
            "what_could_go_wrong": [
                "Scaling on weak confidence can lead to unpredictable results and wasted spend.",
                "You may misread noise as signal and make irreversible budget moves.",
            ],
            "what_to_do_next": [
                "Hold allocations and collect more stable daily data.",
                "Improve tracking / conversion signal quality if possible, then re-run.",
            ],
        },
    }
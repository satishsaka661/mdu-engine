# mdu_engine/portfolio_decision.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Literal, Tuple
import math

Decision = Literal["SCALE", "HOLD", "REDUCE"]
RiskProfile = Literal["conservative", "balanced", "growth"]

# -----------------------------
# Inputs (from your per-platform engine)
# -----------------------------

@dataclass(frozen=True)
class ChannelDecision:
    platform: str                       # "meta" or "google" (or future channels)
    decision: Decision                  # SCALE / HOLD / REDUCE
    avg_net_value: float                # average daily net_value (or window avg)
    downside_risk: float                # higher = worse (must be consistent scale across channels)
    confidence: float                   # 0..1
    spend_total: float                  # total spend in the analyzed window
    notes: Tuple[str, ...] = ()         # optional warnings / quality notes


# -----------------------------
# Portfolio output
# -----------------------------

@dataclass(frozen=True)
class ReallocationRecommendation:
    enabled: bool
    from_platform: Optional[str] = None
    to_platform: Optional[str] = None
    amount: float = 0.0
    expected_downside_risk_reduction_pct: float = 0.0  # relative reduction vs. source channel risk
    expected_confidence_change: float = 0.0            # + means portfolio confidence improves
    rationale: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PortfolioDecision:
    channels: Dict[str, ChannelDecision]               # key: platform
    portfolio_action: str                              # human string summary
    recommendation: ReallocationRecommendation
    portfolio_confidence: float                        # 0..1
    rationale_blocks: Dict[str, Tuple[str, ...]]       # for Step 2 later: happened / risk / next


# -----------------------------
# Core logic
# -----------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _risk_weight(profile: RiskProfile) -> float:
    """
    Higher means prioritize risk reduction more.
    """
    return {
        "conservative": 0.70,
        "balanced": 0.50,
        "growth": 0.35,
    }[profile]


def _min_confidence(profile: RiskProfile) -> float:
    """
    Minimum confidence required to recommend reallocations.
    """
    return {
        "conservative": 0.70,
        "balanced": 0.60,
        "growth": 0.50,
    }[profile]


def _max_shift_pct(profile: RiskProfile) -> float:
    """
    Maximum portion of 'from' channel spend to reallocate in one move.
    """
    return {
        "conservative": 0.15,
        "balanced": 0.20,
        "growth": 0.30,
    }[profile]


def _decision_rank(d: Decision) -> int:
    """
    SCALE > HOLD > REDUCE
    """
    return {"REDUCE": 0, "HOLD": 1, "SCALE": 2}[d]


def _score_channel_for_increase(ch: ChannelDecision, profile: RiskProfile) -> float:
    """
    Score for receiving more budget (higher is better).
    Rule-based blend of:
      - decision intent (SCALE > HOLD > REDUCE)
      - confidence (higher better)
      - downside risk (lower better)
      - avg_net_value (higher better)
    """
    rw = _risk_weight(profile)

    # Normalize-ish transforms (keeps it stable without needing population stats)
    conf = _clamp(ch.confidence, 0.0, 1.0)
    decision_bonus = _decision_rank(ch.decision) / 2.0  # 0.0, 0.5, 1.0

    # Risk: convert to "goodness" by inverse; add epsilon to avoid div by zero
    risk_goodness = 1.0 / (1e-9 + max(0.0, ch.downside_risk))

    # Net value goodness: use tanh to dampen extremes
    value_goodness = math.tanh(ch.avg_net_value / (1e-9 + 1.0))

    # Blend: emphasize confidence + decision, then risk/value depending on profile
    score = (
        0.40 * conf +
        0.25 * decision_bonus +
        (0.25 * rw) * risk_goodness +
        (0.10 * (1 - rw)) * value_goodness
    )
    return float(score)


def _portfolio_confidence(channels: Dict[str, ChannelDecision]) -> float:
    """
    Portfolio confidence should be conservative: the system is only as confident as
    the weakest major channel decision.
    Use spend-weighted confidence with a mild penalty for very low spend.
    """
    total_spend = sum(max(0.0, c.spend_total) for c in channels.values())
    if total_spend <= 0:
        return 0.0

    weighted = 0.0
    for c in channels.values():
        w = max(0.0, c.spend_total) / total_spend
        weighted += w * _clamp(c.confidence, 0.0, 1.0)

    # mild penalty if any channel confidence is very low
    min_conf = min(_clamp(c.confidence, 0.0, 1.0) for c in channels.values())
    penalty = 0.10 * (1.0 - min_conf)

    return _clamp(weighted - penalty, 0.0, 1.0)


def recommend_portfolio_action(
    channels: Dict[str, ChannelDecision],
    risk_profile: RiskProfile = "balanced",
    min_portfolio_confidence: float = 0.60,
    min_signal_separation: float = 500.0,
    max_allowed_downside_risk: float = 0.55,
) -> PortfolioDecision:
    """
    Main entrypoint.
    Expects dict containing at least 2 channels (e.g., meta + google),
    already analyzed by the single-channel engine.
    """
    if len(channels) < 2:
        raise ValueError("Portfolio decision requires at least two channels.")

    port_conf = _portfolio_confidence(channels)
    min_req = _min_confidence(risk_profile)
        # -----------------------------
    # User-controlled portfolio gates (industry standard)
    # -----------------------------
    effective_min_conf = max(float(min_req), float(min_portfolio_confidence))

    if port_conf < effective_min_conf:
     return PortfolioDecision(
        channels=channels,
        portfolio_action="HOLD",
        recommendation=ReallocationRecommendation(
            enabled=False,
            from_platform=None,
            to_platform=None,
            amount=0.0,
            expected_downside_risk_reduction_pct=0.0,
            expected_confidence_change=0.0,
            rationale=(
                f"Portfolio confidence {port_conf:.2f} is below required threshold {effective_min_conf:.2f}.",
            ),
        ),
        portfolio_confidence=float(port_conf),
        rationale_blocks={
            "what_happened": (f"Portfolio confidence computed as {port_conf:.2f}.",),
            "what_could_go_wrong": ("Reallocating with low confidence can increase risk.",),
            "what_to_do_next": ("Collect more days of data or improve signal reliability and re-run.",),
        },
    )

    # Score channels for receiving more budget
    scored = sorted(
        ((p, _score_channel_for_increase(c, risk_profile)) for p, c in channels.items()),
        key=lambda x: x[1],
        reverse=True,
    )

    best_platform, best_score = scored[0]
    worst_platform, worst_score = scored[-1]

    best = channels[best_platform]
    worst = channels[worst_platform]

    # Default: no reallocation unless conditions are met
    rec = ReallocationRecommendation(enabled=False, rationale=())
    portfolio_action = "No cross-channel reallocation recommended yet."
    happened: Tuple[str, ...] = ()
    risk_block: Tuple[str, ...] = ()
    next_block: Tuple[str, ...] = ()

    # Conditions to reallocate:
    # 1) best channel should not be REDUCE
    # 2) worst channel should not be SCALE
    # 3) best confidence should be high enough
    # 4) portfolio confidence should clear threshold
    # 5) best score should exceed worst score by a margin (avoid churn)
    score_margin = best_score - worst_score
    margin_gate = 0.12  # tuned for stability; adjust after you see real data

    can_reallocate = (
        best.decision in ("SCALE", "HOLD") and
        worst.decision in ("HOLD", "REDUCE") and
        best.confidence >= min_req and
        port_conf >= (min_req - 0.05) and
        score_margin >= margin_gate and
        worst.spend_total > 0
    )

    # Determine amount
    if can_reallocate:
        shift_pct = _max_shift_pct(risk_profile)

        # If worst is REDUCE and best is SCALE, allow slightly stronger move
        if worst.decision == "REDUCE" and best.decision == "SCALE":
            shift_pct = min(shift_pct + 0.05, 0.35)

        amount = worst.spend_total * shift_pct

        # Expected downside risk reduction (% of worst risk)
        # If risks are on consistent scale: compare directly.
        if worst.downside_risk > 0:
            risk_reduction_pct = _clamp(
                (worst.downside_risk - best.downside_risk) / worst.downside_risk,
                -1.0, 1.0
            ) * 100.0
        else:
            risk_reduction_pct = 0.0

        # Confidence change: heuristic (move budget from low-confidence to high-confidence)
        conf_change = (best.confidence - worst.confidence) * shift_pct

        rec = ReallocationRecommendation(
            enabled=True,
            from_platform=worst.platform,
            to_platform=best.platform,
            amount=float(max(0.0, amount)),
            expected_downside_risk_reduction_pct=float(risk_reduction_pct),
            expected_confidence_change=float(conf_change),
            rationale=(
                f"{best.platform} is comparatively stronger for additional capital (score gap {score_margin:.2f}).",
                f"{best.platform} decision={best.decision}, confidence={best.confidence:.2f}, downside_risk={best.downside_risk:.3f}.",
                f"{worst.platform} decision={worst.decision}, confidence={worst.confidence:.2f}, downside_risk={worst.downside_risk:.3f}.",
            ),
        )

        portfolio_action = (
            f"Reallocate {amount:,.0f} from {worst.platform} → {best.platform} "
            f"to reduce risk and improve decision stability."
        )

        happened = (
            f"Across channels, {best.platform} shows stronger allocation-readiness than {worst.platform}.",
            f"Score gap indicates a meaningful difference in risk-adjusted stability (Δ={score_margin:.2f}).",
        )
        risk_block = (
            f"Shifting budget may reduce downside risk by ~{risk_reduction_pct:.1f}% (relative to {worst.platform}).",
            f"Portfolio confidence expected change: {conf_change:+.3f}.",
        )
        next_block = (
            f"Move a capped portion ({shift_pct*100:.0f}%) of spend from {worst.platform} to {best.platform}.",
            "Re-run after fresh data (next 7 days) to confirm stability persists before further movement.",
        )
    else:
        # Explain why no move
        reasons = []
        if port_conf < (min_req - 0.05):
            reasons.append(f"Portfolio confidence too low ({port_conf:.2f}) for reallocation under {risk_profile}.")
        if best.confidence < min_req:
            reasons.append(f"Top channel confidence below threshold ({best.confidence:.2f} < {min_req:.2f}).")
        if score_margin < margin_gate:
            reasons.append(f"Signal separation too small (score gap {score_margin:.2f} < {margin_gate:.2f}).")
        if worst.decision == "SCALE":
            reasons.append(f"Lowest-ranked channel is still SCALE; reallocation may be premature.")
        if best.decision == "REDUCE":
            reasons.append(f"Highest-ranked channel is REDUCE; portfolio is not in a scaling posture.")

        happened = ("Cross-channel signals are not separated enough to justify moving capital.",)
        risk_block = tuple(reasons) if reasons else ("No strong cross-channel risk advantage detected.",)
        next_block = ("Hold allocations; collect more stable signal before reallocating.",)

        rec = ReallocationRecommendation(enabled=False, rationale=tuple(reasons))

    return PortfolioDecision(
        channels=channels,
        portfolio_action=portfolio_action,
        recommendation=rec,
        portfolio_confidence=port_conf,
        rationale_blocks={
            "what_happened": happened,
            "what_could_go_wrong": risk_block,
            "what_to_do_next": next_block,
        },
    )

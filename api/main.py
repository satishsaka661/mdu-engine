"""
MDU Engine API v1
Decision-support REST API for data-driven optimisation.
Built on FastAPI. Deployed on Google Cloud Run.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import hashlib
from datetime import datetime, timezone

from mdu_engine.decision_confidence import compute_decision_confidence
from mdu_engine.decision_rules import RISK_PROFILES, decide_action
from mdu_engine.version import ENGINE_VERSION, RULESET_VERSION
from mdu_engine.validation import validate_normalized_daily_schema, validation_to_dict

# ── App setup ────────────────────────────────────────────
app = FastAPI(
    title="MDU Engine API",
    description="""
## MDU Engine — Decision Support API

A fail-closed decision-support API for paid media optimisation.

### What it does
Upload daily performance data and receive a structured decision outcome:
**SCALE / HOLD / REDUCE / BLOCK** — with confidence tier, primary constraint,
and a reproducible audit snapshot.

### Design principles
- **Fail-closed**: defaults to HOLD or BLOCK under uncertainty
- **Explainable**: every outcome includes the primary constraint and rationale
- **Auditable**: every response includes a versioned, reproducible snapshot
- **Human-in-the-loop**: provides guidance only, never executes changes

### Important
MDU Engine provides decision support only. All decisions remain the
responsibility of the operator. This API does not execute budget changes
or provide financial advice.

**Built by Satish Saka** · [mduengine.com](https://mduengine.com) · MIT License
    """,
    version=ENGINE_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────
class DailyRecord(BaseModel):
    date: str = Field(..., example="2026-03-01", description="Date in YYYY-MM-DD format")
    spend: float = Field(..., example=1250.50, description="Total spend for this day")
    conversions: float = Field(..., example=42.0, description="Total conversions for this day")
    value_per_conversion: float = Field(default=0.0, example=5000.0, description="Value per conversion in your currency")
    net_value: Optional[float] = Field(default=None, example=210000.0, description="Net value (conversions × value_per_conversion − spend). Auto-calculated if not provided.")


class DecideRequest(BaseModel):
    data: list[DailyRecord] = Field(
        ...,
        description="List of daily performance records. Minimum 7 days recommended, maximum 30 days.",
        min_length=1,
    )
    risk_profile: str = Field(
        default="balanced",
        description="Risk profile to apply. Options: balanced, conservative, growth.",
        example="balanced",
    )
    signal_reliability: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Signal reliability score between 0 and 1. Use 0.6 if unsure.",
        example=0.6,
    )
    scale_pct: float = Field(
        default=0.10,
        ge=0.01,
        le=0.50,
        description="Proposed scale percentage as a decimal. 0.10 = 10% scale.",
        example=0.10,
    )
    simulations: int = Field(
        default=5000,
        description="Number of Monte Carlo simulations. Higher = more accurate but slower.",
        example=5000,
    )
    platform: str = Field(
        default="unknown",
        description="Platform label for audit purposes.",
        example="Meta Ads",
    )


class DecisionResponse(BaseModel):
    status: str
    outcome: str
    confidence_tier: str
    primary_constraint: str
    recommended_change_pct_range: str
    next_review_window: str
    reason: str
    decision_mode: str
    days_of_data: int
    date_min: Optional[str]
    date_max: Optional[str]
    spend_total: float
    decision_confidence: float
    downside_risk: float
    engine_version: str
    ruleset_version: str
    random_seed: int
    input_hash: str
    generated_at_utc: str
    platform: str


# ── Helper functions ──────────────────────────────────────
def stable_seed(df: pd.DataFrame) -> int:
    df2 = df.copy()
    cols = [c for c in ["date", "spend", "conversions", "value_per_conversion", "net_value"] if c in df2.columns]
    df2 = df2[cols].copy()
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce").dt.date.astype(str)
    for c in cols:
        if c != "date":
            df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0.0).round(6)
    payload = df2.to_csv(index=False).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def input_hash(df: pd.DataFrame) -> str:
    df2 = df.copy()
    cols = [c for c in ["date", "spend", "conversions", "value_per_conversion", "net_value"] if c in df2.columns]
    df2 = df2[cols].copy()
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce").dt.date.astype(str)
    for c in cols:
        if c != "date":
            df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0.0).round(6)
    payload = df2.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def classify_constraint(status: str, action: str) -> str:
    if status == "DECISION_BLOCKED":
        return "Data Sufficiency Gate"
    if action == "REDUCE":
        return "Downside Risk Gate"
    return "Confidence Gate"


def decision_mode(status: str, action: str) -> str:
    if status == "DECISION_BLOCKED":
        return "Epistemic (refusal under uncertainty)"
    if action in ("HOLD", "REDUCE"):
        return "Strategic (restraint under known risk)"
    if action == "SCALE":
        return "Permissive (action allowed under constraints)"
    return "Strategic (restraint under known risk)"


# ── Routes ────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health_check():
    """
    Health check endpoint. Returns engine status and version information.
    """
    return {
        "status": "ok",
        "service": "MDU Engine API",
        "engine_version": ENGINE_VERSION,
        "ruleset_version": RULESET_VERSION,
        "docs": "/docs",
        "product": "https://mduengine.com",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/decide", response_model=DecisionResponse, tags=["Decision"])
def decide(request: DecideRequest):
    """
    ## Core Decision Endpoint

    Submit daily performance data and receive a structured decision outcome.

    ### Input
    A list of daily records with date, spend, conversions, and optionally
    value_per_conversion and net_value.

    ### Output
    A decision outcome (SCALE / HOLD / REDUCE / BLOCK) with:
    - Confidence tier (High / Medium / Low / n/a)
    - Primary constraint (what is limiting the decision)
    - Recommended budget change range
    - Full audit metadata (seed, hash, versions)

    ### Notes
    - Minimum 7 days of data recommended
    - Maximum 30 days for best results
    - HOLD and BLOCK are valid outcomes, not failures
    - The system never executes changes — guidance only
    """

    # Validate risk profile
    if request.risk_profile not in RISK_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid risk_profile. Must be one of: {list(RISK_PROFILES.keys())}"
        )

    profile = RISK_PROFILES[request.risk_profile]

    try:
        # Build normalised dataframe from request
        records = []
        for r in request.data:
            net_val = r.net_value
            if net_val is None:
                net_val = (r.conversions * r.value_per_conversion) - r.spend
            records.append({
                "date": r.date,
                "spend": r.spend,
                "conversions": r.conversions,
                "value_per_conversion": r.value_per_conversion,
                "net_value": net_val,
            })

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        if len(df) == 0:
            raise HTTPException(status_code=400, detail="No valid date records found after parsing.")

        # Validate
        from mdu_engine.validation import validate_normalized_daily_schema, validation_to_dict
        v = validate_normalized_daily_schema(df)

        seed = stable_seed(df)
        ihash = input_hash(df)
        date_min = str(df["date"].min().date()) if len(df) else None
        date_max = str(df["date"].max().date()) if len(df) else None
        days = len(df)
        spend_total = float(df["spend"].sum())

        if not v.is_valid:
            return DecisionResponse(
                status="DECISION_BLOCKED",
                outcome="BLOCK",
                confidence_tier="n/a",
                primary_constraint="Data Sufficiency Gate",
                recommended_change_pct_range="0% (no change)",
                next_review_window="After fixing data",
                reason=v.block_reason or "Data validation failed",
                decision_mode="Epistemic (refusal under uncertainty)",
                days_of_data=days,
                date_min=date_min,
                date_max=date_max,
                spend_total=spend_total,
                decision_confidence=0.0,
                downside_risk=1.0,
                engine_version=ENGINE_VERSION,
                ruleset_version=RULESET_VERSION,
                random_seed=seed,
                input_hash=ihash,
                generated_at_utc=datetime.now(timezone.utc).isoformat(),
                platform=request.platform,
            )

        # Monte Carlo
        mc = compute_decision_confidence(
            df,
            signal_reliability=request.signal_reliability,
            scale_pct=request.scale_pct,
            simulations=request.simulations,
        )

        decision = decide_action(
            decision_confidence=mc["decision_confidence"],
            downside_risk=mc["downside_risk"],
            profile=profile,
            days_of_data=days,
        )

        status = decision.get("status", "DECISION_OK")
        action = decision.get("action", "HOLD")

        conf = float(mc.get("decision_confidence", 0.0))
        if status == "DECISION_BLOCKED":
            conf_tier = "n/a"
        elif conf >= 0.75:
            conf_tier = "High"
        elif conf >= 0.50:
            conf_tier = "Medium"
        else:
            conf_tier = "Low"

        return DecisionResponse(
            status=status,
            outcome="BLOCK" if status == "DECISION_BLOCKED" else action,
            confidence_tier=conf_tier,
            primary_constraint=classify_constraint(status, action),
            recommended_change_pct_range=decision.get("recommended_change_pct_range", "0%"),
            next_review_window=decision.get("next_review_window", "72 hours"),
            reason=decision.get("reason", ""),
            decision_mode=decision_mode(status, action),
            days_of_data=days,
            date_min=date_min,
            date_max=date_max,
            spend_total=spend_total,
            decision_confidence=conf,
            downside_risk=float(mc.get("downside_risk", 1.0)),
            engine_version=ENGINE_VERSION,
            ruleset_version=RULESET_VERSION,
            random_seed=seed,
            input_hash=ihash,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            platform=request.platform,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Decision engine error: {str(e)}"
        )
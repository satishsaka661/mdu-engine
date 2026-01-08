## Release Contract (v1.0.0)

This release defines the first stable public contract of MDU Engine.

Guarantees:
- Input CSV schema validation before any decision is made
- Deterministic Monte Carlo simulations via seeded randomness
- Explicit decision confidence and downside risk reporting
- Explainable outputs for all recommendations
- No automatic budget changes — advisory only

Breaking changes will only be introduced in major releases (v2.0.0+).

## How to Trust the Output

MDU Engine does not rely on black-box optimization.

Trust is established through:
- Explicit data validation (daily granularity, sufficient window)
- Monte Carlo simulations instead of point estimates
- Confidence tiers rather than binary decisions
- Downside-risk-aware recommendations
- Full visibility into assumptions and constraints

If validation fails, the engine blocks decisions by design.

## What makes MDU Engine trustworthy?

- Deterministic Monte Carlo simulations (same data → same result)
- Explicit data validation before decisions
- Versioned engine and rulesets
- Human-readable decision explanations
- Persistent audit history (last N decisions)
- Open-source MIT license

## Decision Contract

MDU Engine guarantees:
1. No decision without valid data
2. No hidden thresholds
3. No mutation of user data
4. Full traceability of decisions

## Known Limitations

- Designed for 7–30 day windows (not intraday)
- Not a forecasting system
- Not a replacement for domain judgment

## What is MDU Engine?

MDU Engine is an open-source decision engine that helps marketers and founders
decide whether to **SCALE**, **HOLD**, or **REDUCE** ad spend across Meta Ads and Google Ads.

It is designed for:
- transparency over black-box automation
- reproducible decision-making
- explicit risk and confidence modeling

MDU Engine is **not an ad platform** and **does not place ads**.
It provides **decision support**, not execution.

## Why MDU Engine is trustworthy

MDU Engine is built around explicit, inspectable decision logic:

- Deterministic Monte Carlo simulations  
  (same input data → same decision outcome)

- Strict data validation  
  (no decisions made on insufficient or malformed data)

- Versioned engine and rulesets  
  (engine logic and decision thresholds are tracked separately)

- Human-readable explanations  
  (every decision explains *what happened*, *what could go wrong*, and *what to do next*)

- Persistent audit history  
  (past decisions are logged with timestamps, confidence, risk, and versions)

- Open-source under the MIT License  
  (no hidden logic or proprietary lock-in)

Decision Guardrails & Validation 

MDU Engine is designed as a decision-support system, not an automated execution tool.
To ensure reliability and prevent unsafe recommendations, the engine applies strict validation guardrails before issuing any decision.

When a Decision Is Blocked

A decision will be intentionally blocked if the uploaded data does not meet minimum reliability requirements, including:

Fewer than 7 daily rows (7–30 days recommended)

Data not segmented by Day

Missing or malformed spend, conversion, or date fields

Aggregated or summary-only reports (monthly totals, campaign-only exports)

When blocked, the engine does not run simulations and defaults to a safe HOLD action.

Example: Insufficient Daily Data Window

If a user uploads data containing only 1 day, MDU Engine responds with:

Action: HOLD

Confidence Tier: n/a

Budget Change: 0%

Reason: Insufficient daily data window

Next Review: After fixing export

The UI also provides structured explainability:

What happened: Insufficient daily data window

What could go wrong: Acting on insufficient data may lead to incorrect budget changes

What to do next: Export a daily report (Breakdown: Day) with at least 7–30 days and re-upload

This behavior is intentional and non-negotiable.

Design Philosophy

MDU Engine prioritizes:

Decision safety over automation

Transparency over black-box recommendations

Human judgment over blind execution

The engine will always refuse to act when confidence is mathematically or statistically unjustified.

Reference Screenshots

Real UI examples demonstrating blocked decisions and validation feedback are available in the repository:

Decision blocked due to insufficient data window

Validation feedback and user guidance

Portfolio decision guardrails

These screenshots reflect the live behavior of the public Streamlit app.

## Decision Contract

MDU Engine follows a strict decision contract:

1. No decision is produced without valid daily data.
2. No thresholds are hidden or adaptive.
3. No user data is mutated or stored externally.
4. Every decision is traceable and explainable.
5. Engine and ruleset versions are always recorded.

This makes MDU Engine suitable for review, auditing, and long-term use.

## What MDU Engine is NOT

- ❌ Not an automated bidding system
- ❌ Not a forecasting or prediction engine
- ❌ Not a replacement for human judgment
- ❌ Not a black-box AI tool

MDU Engine is a **decision-support system**, designed to augment human reasoning,
not replace it.

## Known Limitations

- Designed for **daily-level data** (7–30 days recommended)
- Not suitable for intraday or real-time bidding decisions
- Portfolio recommendations are conservative by design
- Results depend on the quality of exported ad platform data

These limitations are intentional to preserve decision integrity.

## Sample Data

This repository includes sample daily datasets to help users explore the MDU Engine without using real ad account data.

**Included samples:**
- Meta Ads (daily): `samples/meta_ads/meta_sample_daily.csv`
- Google Ads (daily): `samples/google_ads/google_sample_daily.csv`

These files follow the normalized schema used by the engine and can be uploaded directly into the Streamlit app.

# MDU Engine — Meta + Google Ads Decision Engine

MDU Engine is a Streamlit-based decision engine that helps marketers make safer budget decisions for Meta Ads and Google Ads using daily performance exports. It normalizes data, validates data quality, runs Monte Carlo simulations, and outputs SCALE / HOLD / REDUCE recommendations with explainability and downloadable reports.

## Live App
- Streamlit: https://mdu-engine-satish-saka.streamlit.app/

## What it supports

### Meta Ads (Daily Export)
Required:
- Breakdown by **Day**
- Columns: Date/Day/Reporting starts, Amount spent, Results

### Google Ads (Daily Export)
Required:
- Segmented by **Day**
- Columns: Date/Day, Cost, Conversions (or All conversions)

## How it works (pipeline)
1. Upload CSV (Meta or Google)
2. Auto-detect platform and normalize into a daily schema:
   - date, spend, conversions, value_per_conversion, net_value
3. Validate data quality (blocks bad exports)
4. Monte Carlo simulation → confidence + downside risk
5. Decision rules (risk profile + data window gating)
6. Explainability + report + history logs
7. Optional cross-channel portfolio reallocation

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
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
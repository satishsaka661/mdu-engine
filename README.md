# MDU Engine — Meta + Google Ads Decision Engine

MDU Engine is an open-source **decision-support engine** that helps marketers and founders decide whether to **SCALE**, **HOLD**, or **REDUCE** ad spend across **Meta Ads** and **Google Ads**, using daily performance exports.

It prioritizes **decision safety, transparency, and reproducibility** over black-box automation.

![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Status](https://img.shields.io/badge/status-stable-brightgreen)

---

## Live App
👉 https://mdu-engine-satish-saka.streamlit.app/

---

## What MDU Engine Does

- Normalizes Meta Ads and Google Ads daily exports
- Validates data quality before making any decision
- Runs deterministic Monte Carlo simulations
- Outputs **SCALE / HOLD / REDUCE** recommendations
- Reports decision confidence and downside risk
- Provides human-readable explainability
- Supports optional cross-channel portfolio reallocation
- Logs decisions for audit and review

---

## What MDU Engine Does NOT Do

- ❌ Does not place ads
- ❌ Does not execute budget changes
- ❌ Does not connect to ad platforms
- ❌ Does not provide financial or investment advice
- ❌ Does not replace human judgment

MDU Engine is **advisory only**.

---

## Release Contract (v1.0.0)

This release defines the first stable public contract of MDU Engine.

### Guarantees
- Input CSV validation before any decision
- Deterministic Monte Carlo simulations
- Explicit confidence and downside risk reporting
- Explainable outputs for all recommendations
- No automatic execution — advisory only

Breaking changes will only occur in **major releases (v2.0.0+)**.

---

## Decision Contract

MDU Engine guarantees:

1. No decision without valid daily data
2. No hidden or adaptive thresholds
3. No mutation of user data
4. Full traceability of every decision
5. Engine and ruleset versions always recorded

---

## Decision Guardrails & Validation

MDU Engine intentionally blocks unsafe decisions.

### A decision is blocked if:
- Fewer than **7 daily rows** (7–30 recommended)
- Data is not segmented by **Day**
- Required fields are missing or malformed
- Data is aggregated (monthly totals, summaries)

When blocked:
- Action defaults to **HOLD**
- No simulations are run
- Clear guidance is shown to the user

This behavior is **intentional and non-negotiable**.

---

## Example: Insufficient Daily Data Window

If only 1 day is uploaded:

- Action: **HOLD**
- Confidence Tier: **n/a**
- Budget Change: **0%**
- Reason: Insufficient daily data window
- Next Review: After fixing export

The UI explains:
- What happened
- What could go wrong
- What to do next

---

## How MDU Engine Establishes Trust

- Deterministic simulations (same data → same result)
- Explicit validation before decisions
- Versioned engine and rulesets
- Human-readable explanations
- Persistent decision history
- Open-source under MIT License

---

## Known Limitations

- Designed for **daily data only**
- Best for **7–30 day windows**
- Not a forecasting system
- Portfolio recommendations are conservative by design

These limitations preserve decision integrity.

---

## How It Works (Pipeline)

1. Upload CSV (Meta or Google)
2. Platform auto-detection and normalization
3. Data validation (blocks bad exports)
4. Monte Carlo simulation
5. Decision rules (risk-aware)
6. Explainability and report generation
7. Optional portfolio reallocation

---

## Sample Data (Safe Evaluation)

Use included sample datasets to evaluate the engine safely.

samples/
├── meta_ads/
│ └── meta_sample_daily.csv
└── google_ads/
└── google_sample_daily.csv

yaml
Copy code

These samples:
- Always pass validation
- Produce deterministic decisions
- Demonstrate portfolio logic

---

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
Reproducibility & Auditability
MDU Engine derives a deterministic random seed from normalized input data.

Same data → same seed
Same seed → same Monte Carlo outcome
Decisions are fully reproducible and auditable

This design supports professional review and governance.

Versioning Policy
MDU Engine follows semantic versioning:

MAJOR (X.0.0) – Breaking decision logic
MINOR (1.X.0) – New capabilities or validations
PATCH (1.0.X) – Fixes and documentation

Every decision records:

Engine version
Ruleset version
Random seed (if applicable)

Disclaimer
MDU Engine is a decision-support system only.

It does not execute changes and does not provide financial or investment advice.

All recommendations must be reviewed and approved by a qualified human decision-maker.

yaml
Copy code

---

# ✅ AFTER YOU PASTE (ONLY THESE COMMANDS)

Yes — **run these in terminal**:

```bash
git add README.md
git commit -m "docs: finalize v1.0 public contract and usage"
git push
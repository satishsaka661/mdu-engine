![MIT License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Status](https://img.shields.io/badge/status-stable-brightgreen)
![Build](https://img.shields.io/badge/build-passing-success)

# MDU Engine — Meta + Google Ads Decision Engine

MDU Engine is an open-source **decision-support system** that helps marketers, analysts, and founders decide whether to **SCALE**, **HOLD**, or **REDUCE** ad spend across **Meta Ads** and **Google Ads** using validated daily performance data.

The engine prioritizes **decision safety, deterministic logic, transparency, explainability, and auditability** over black-box automation.

![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Status](https://img.shields.io/badge/status-stable-brightgreen)

---

## Live App

https://mdu-engine-satish-saka.streamlit.app/

---

## What MDU Engine Does

- Normalizes daily ad exports from Meta Ads and Google Ads  
- Validates data quality **before** any decision  
- Runs deterministic Monte Carlo simulations  
- Reports explicit **decision confidence** and **downside risk**  
- Produces **SCALE / HOLD / REDUCE** recommendations  
- Provides human-readable explanations  
- Supports portfolio-level reallocation  
- Logs decisions for audit and review  

---

## What MDU Engine Does NOT Do

- ❌ No automated budget execution  
- ❌ No ad platform integrations  
- ❌ No forecasting or prediction claims  
- ❌ No black-box AI logic  
- ❌ No financial advice  

MDU Engine is **advisory only**.

---

## Release Contract — v1.0.0

This release defines the first stable public contract of MDU Engine.

### Guarantees

- Strict input CSV validation  
- Deterministic Monte Carlo via seeded randomness  
- Explicit confidence and downside risk  
- Explainable outputs  
- Advisory-only recommendations  
- Breaking changes only in v2.0.0+  

---

## Decision Contract

MDU Engine guarantees:

1. No decision without valid daily data  
2. No hidden thresholds  
3. No mutation of user data  
4. Full traceability  
5. Versioned engine and rulesets  

---

## Decision Guardrails & Validation

A decision is **blocked** if:

- Fewer than **7 daily rows**  
- Data is not segmented by **Day**  
- Spend, conversions, or dates are missing  
- Aggregated / summary exports are uploaded  

When blocked:

- Action: HOLD  
- Confidence: n/a  
- Budget Change: 0%  
- Simulations: Not executed  

This behavior is intentional.

---

## Reproducibility & Auditability

- Same input → same seed  
- Same seed → same Monte Carlo output  
- Every decision logs engine version, ruleset version, seed, risk, and confidence  

---

## Engine Pipeline

1. Upload CSV  
2. Detect platform & normalize  
3. Validate schema & data window  
4. Run Monte Carlo simulation  
5. Apply decision rules  
6. Generate explainability & report  
7. Optional portfolio reallocation  

---

## Sample Data
samples/
├── meta_ads/
│ └── meta_sample_daily.csv
└── google_ads/
└── google_sample_daily.csv


Sample files always pass validation and produce deterministic results.

---

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

## ⚠️ Production Use Disclaimer

MDU Engine is a decision-support system.

• It does not execute changes  
• It does not connect to ad platforms  
• It does not spend money  
• It does not provide financial advice  

All outputs must be reviewed by a qualified human decision-maker before being acted upon.


# MDU Engine — Decision Report

**Generated:** 2026-01-08 11:36:41

**Platform:** Google Ads

**Risk Profile:** balanced

**Engine Version:** 1.0.0

**Ruleset Version:** 2026-01-08

**Random Seed:** None

**Validation Status:** DECISION_OK


---

## Recommendation Summary

```
=== Recommendation Summary ===
risk_profile: balanced
action: REDUCE
confidence: 0.243 (LOW)
downside_risk: 0.595 (HIGH)
why: Risk is too high right now. Reduce budget and diagnose issues (tracking, creatives, audience, landing page) before scaling again.
```

---

## Decision Details

- action: REDUCE

- reason: Downside risk is extreme (0.595 ≥ 0.55). Reduce spend and investigate.

- user_explanation: Risk is too high right now. Reduce budget and diagnose issues (tracking, creatives, audience, landing page) before scaling again.


---

## Why this decision?

### What happened

- Downside risk spiked to 0.595, above the REDUCE threshold for balanced.

- The system sees a high chance of negative net outcomes at current allocation.


### What could go wrong

- Continuing current spend may amplify losses while the signal is unstable.

- If tracking/conversion quality is broken, decisions will be misleading.


### What to do next

- Reduce spend immediately to control downside exposure.

- Validate tracking and conversion integrity; fix data quality issues.

- Re-run after 7 more days of stable data before scaling again.


---

## Data Window

- date_range: 2024-03-01 → 2024-03-15

- days_of_data: 15


---

## Raw Metrics

- decision_confidence: 0.243

- downside_risk: 0.595

- avg_net_value: -2013.42

- spend_total: 30201.239999999998

- simulations: 5000

- signal_reliability: 0.6

- scale_pct: 0.1

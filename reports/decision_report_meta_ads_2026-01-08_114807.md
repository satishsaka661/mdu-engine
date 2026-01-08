# MDU Engine — Decision Report

**Generated:** 2026-01-08 11:48:07

**Platform:** Meta Ads

**Risk Profile:** balanced

**Engine Version:** 1.0.0

**Ruleset Version:** 2026-01-08

**Random Seed:** n/a

**Validation Status:** DECISION_BLOCKED

**Block Reason:** Insufficient daily data window: 1 day(s). Provide at least 7 daily rows (7–30 recommended).


---

## Recommendation Summary

```
=== Recommendation Summary ===
risk_profile: balanced
action: HOLD
confidence: 0.000 (LOW)
downside_risk: 1.000 (HIGH)
why: I can’t make a reliable scale/reduce decision because the export/data window is not suitable. Fix the export (daily breakdown, 7–30 days) and re-upload.
```

---

## Decision Details

- action: HOLD

- reason: Insufficient daily data window: 1 day(s). Provide at least 7 daily rows (7–30 recommended).

- user_explanation: I can’t make a reliable scale/reduce decision because the export/data window is not suitable. Fix the export (daily breakdown, 7–30 days) and re-upload.


---

## Why this decision?

### What happened

- Insufficient daily data window: 1 day(s). Provide at least 7 daily rows (7–30 recommended).


### What could go wrong

- Acting on insufficient or malformed data can lead to incorrect budget changes.


### What to do next

- Export a daily report (Breakdown: Day) with at least 7 days (7–30 recommended) and re-upload.


---

## Data Window

- date_range: 2025-12-01 → 2025-12-01

- days_of_data: 1


---

## Raw Metrics

- decision_confidence: 0.0

- downside_risk: 1.0

- avg_net_value: 1246779.6

- spend_total: 43220.4

- simulations: 5000

- signal_reliability: 0.6

- scale_pct: 0.1

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
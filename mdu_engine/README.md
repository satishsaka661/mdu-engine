# MDU Engine

MDU Engine is a Streamlit-based decision engine for Meta Ads and Google Ads.

It helps decide whether to SCALE, HOLD, or REDUCE ad spend using:
- Daily performance data
- Monte Carlo simulation
- Risk-aware decision rules

## What it does
1. Accepts Meta Ads or Google Ads CSV exports
2. Normalizes data into a daily format
3. Validates data quality (blocks bad exports)
4. Estimates decision confidence and downside risk
5. Produces a clear action with explanation
6. Generates a Markdown decision report
7. Logs all decisions for audit and review

## How to run
```bash
pip install -r requirements.txt
streamlit run app.py
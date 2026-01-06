# FOMC Influence Validation Tool

A Streamlit application for validating LLM-extracted influence scores from Federal Reserve FOMC meeting transcripts.

## Purpose

This tool validates Claude's assessments of how much influence FOMC meeting participants had on policy decisions. Human coders review Claude's influence scores and justifications, then provide their own assessments.

## Target Meetings

| Date | Context |
|------|---------|
| October 6, 1979 | Volcker's Saturday Night Special |
| August 16, 1994 | Greenspan tightening cycle |
| December 16, 2008 | Financial crisis ZLB |
| August 9, 2011 | Calendar guidance introduced |
| July 31, 2019 | Powell mid-cycle cut |

## Influence Scale

| Score | Meaning | Criteria |
|-------|---------|----------|
| 0 | No influence | Did not discuss, opposed, or contradicted decision |
| 1 | Limited influence | Mentioned topic, general support, no substantive arguments |
| 2 | Moderate influence | Provided arguments, discussed details, defended policy |
| 3 | Strong influence | Proposed adopted policy, unique central arguments, shaped decision |

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data Files

- `data/policy_influence_vF.pkl` - Claude's influence assessments
- `data/adopted_decisions.pkl` - FOMC decisions
- `data/transcripts.parquet` - Meeting transcripts
- `data/alternatives.pkl` - Policy alternatives

# Alert Agent — Custom BaseAgent

## What It Does
Real-time safety monitoring with 5-factor scoring system. Dispatches SMS to guardian when crisis threshold is met. Runs in parallel with the main response pipeline.

## 5 Factors
- F1: Crisis keyword in transcript
- F2: Sustained negative emotion (high confidence)
- F3: Emotional escalation pattern (3+ consecutive)
- F4: Vulnerability (pre-set True for minors)
- F5: Explicit self-harm statement

## Threshold
- General: 3-of-5 factors
- Minors: F4 pre-set True → effectively 2-of-4

## ADK Type
Custom `BaseAgent` with `_run_async_impl`

## Env Vars
- `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`
- `DATABASE_URL`, `REDIS_URL`

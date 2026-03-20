# BEAN AI — Safety System

BEAN serves teenagers aged 13–17. The safety system is a core part of the product, not an afterthought. This document describes how it works, why it was designed this way, and what happens when it triggers.

---

## Overview

Every single conversation turn passes through the safety system. The `AlertAgent` runs **in parallel** with memory retrieval on every turn — it never blocks the response pipeline, but it can append a post-alert message to the response if a threshold is crossed.

The system uses a **5-factor scoring model**: an alert is dispatched when enough independent factors are present simultaneously. This avoids both false positives (single keyword triggers) and false negatives (no single factor alone, but a concerning pattern overall).

---

## The 5 Factors

### F1 — Crisis Keyword

A keyword or phrase from the crisis keyword list is detected in the transcript.

Adults use `CRISIS_KEYWORDS_GENERAL`. Minors use `CRISIS_KEYWORDS_MINOR`, which is a superset that includes additional vulnerability signals:

**General keywords include:** explicit self-harm and suicide statements (`"kill myself"`, `"want to die"`, `"end my life"`, `"overdose"`, etc.)

**Minor-only additions include:** isolation signals (`"nobody cares"`, `"no one would miss me"`), bullying/abuse disclosures (`"being bullied"`, `"he hits me"`, `"touched me"`, `"abused"`), eating disorder language (`"starving myself"`, `"purging"`), and self-harm variations (`"cutting"`, `"burning myself"`).

---

### F2 — Sustained Negative Emotion

The wav2vec2 emotion model returns a high-confidence negative emotion label.

- Labels that trigger F2: `angry`, `fearful`, `sad`, `disgusted`
- Confidence threshold: configurable (default high — low-confidence detections do not trigger)
- A single sad detection does not trigger F2; it requires sustained, high-confidence negative affect

---

### F3 — Emotional Escalation Pattern

The recent emotion history (stored in session state as a rolling window) shows a pattern of escalating negative affect across multiple turns. A single upset moment does not trigger F3 — it requires a trajectory of worsening emotion over the conversation.

---

### F4 — Vulnerability (Minor)

**Pre-set to `True` for all minor users before any evaluation.**

This is the most important minor-specific mechanism. For a user flagged as `is_minor=True`:

- F4 is automatically active at the start of every turn
- The effective alert threshold drops from 3-of-5 to 2-of-4 (since F4 is always counted)
- A minor only needs **one additional factor** (e.g. F1 crisis keyword alone) to trigger an alert
- Adults require 3 independent factors

This reflects the product's duty of care: minors are an inherently vulnerable population, and the cost of a missed alert is higher than the cost of a false positive.

---

### F5 — Explicit Self-Harm Statement

A high-severity explicit statement is detected — not just crisis-adjacent language, but direct, actionable statements:

Examples: `"I'm going to kill myself"`, `"I have a plan"`, `"tonight is the night"`, `"I wrote a note"`, `"I have the pills"`, `"goodbye forever"`

F5 is treated as the highest-severity signal. In practice, F5 alone in combination with F4 (minor) immediately triggers an alert.

---

## Alert Thresholds

| User type | Threshold | Notes |
|-----------|-----------|-------|
| Adult | 3 of 5 factors | Standard threshold |
| Minor | 2 of 4 factors | F4 always pre-set; effectively 1 additional factor needed |

Thresholds are configurable via `ALERT_THRESHOLD` and `MINOR_ALERT_THRESHOLD` env vars.

---

## Alert Levels

The `AlertState.compute_level()` method maps factor counts to alert levels:

| Level | Factors active | Action |
|-------|---------------|--------|
| `NONE` | Below threshold | No action |
| `ELEVATED` | Approaching threshold | Logged only |
| `HIGH` | At threshold | SMS dispatched |
| `CRISIS` | F5 present | SMS dispatched immediately |

---

## What Happens When an Alert Triggers

```
1. AlertAgent calls evaluate_factors() → AlertState with level HIGH or CRISIS

2. Alert state persisted to Redis (keyed by session_id)
   → Prevents duplicate alerts within the same session

3. send_guardian_alert() called via Twilio SMS
   → Sends to guardian phone number stored in users table
   → SMS includes session context (sanitised — no raw transcript)
   → Twilio SID logged to alert_logs table

4. session.state["alert_dispatched"] = "true"
   session.state["post_alert_message"] = <transparent message>

5. Orchestrator appends post_alert_message to the response:

   "I want you to know that because I care about your safety, a guardian
   has been notified. You're not in trouble — I just want to make sure
   you have the support you need. Would you like to keep talking?"

6. BEAN continues the conversation warmly and supportively
   → The response agent (therapy or casual) still generates a response
   → The user is not cut off, interrogated, or alarmed
```

**The post-alert message is always transparent.** BEAN never pretends an alert didn't happen. This is intentional — deception would erode trust with the user and is inconsistent with BEAN's character.

---

## State Persistence (Redis)

Alert state is persisted in Redis keyed by `session_id`. This serves two purposes:

1. **Factor accumulation across turns** — F2 (sustained emotion) and F3 (escalation pattern) require history. State carries forward turn-to-turn within a session.
2. **Deduplication** — Once `alert_dispatched=true` is set, the AlertAgent skips SMS dispatch on subsequent turns in the same session, preventing spam to the guardian.

Alert state is cleared when the session ends (`session_cleanup` background worker).

---

## Alert Logs

Every alert dispatch is written to the `alert_logs` table with:
- `session_id`, `user_id`
- Active factors at time of dispatch
- Alert level
- Twilio message SID (or error if SMS failed)
- Timestamp

Accessible via `GET /api/alerts` (guardian/admin auth required).

---

## SafetyService in TherapeuticConvoAgent

The `TherapeuticConvoAgent` has a `before_model_callback` that runs `check_crisis_keywords()` and `check_explicit_statement()` **before** Gemini generates a response. This is a second, inline safety check specifically for the therapy path:

- If F1 or F5 keywords are detected, the callback can modify the prompt to ensure BEAN's response is appropriate
- This does not replace the AlertAgent — it runs in addition to it
- Ensures the therapy response itself doesn't inadvertently say something harmful even before the turn-level alert fires

---

## Limitations and Known Gaps

- **Audio-only emotion detection** — wav2vec2 runs on audio, not text. Text-only sessions (if ever added) would have no F2/F3 signals.
- **Keyword matching is literal** — obfuscated language (`"k*ll myself"`, coded terms) is not caught by F1/F5. A future improvement would be LLM-based semantic crisis detection as a fallback.
- **No real-time guardian dashboard** — guardians currently receive only SMS. A companion app with alert history is a planned feature.
- **Single guardian contact** — the system supports one guardian phone number per user. Multi-guardian support is not yet implemented.
- **No escalation to emergency services** — the system notifies guardians but does not call 911 or crisis lines. This is intentional (avoiding legal/regulatory complexity) but should be revisited.

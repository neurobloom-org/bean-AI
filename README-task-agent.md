# Task Agent — LlmAgent + Calendar FunctionTools

## What It Does
Manages tasks, reminders, and calendar events via Google Calendar API. Creates "BEAN Reminders" calendar on first use. Supports create, list, cancel, and snooze operations.

## Tools
- `create_calendar_event`: Create event + task record
- `list_calendar_events`: List upcoming events
- `cancel_calendar_event`: Cancel event + update task
- `snooze_reminder`: Push reminder forward

## ADK Type
`LlmAgent` (Gemini 2.5 Flash) with Calendar `FunctionTools`

## Env Vars
- `GOOGLE_API_KEY`, `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`

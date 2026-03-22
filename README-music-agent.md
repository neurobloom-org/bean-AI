# Music Agent — LlmAgent + Music FunctionTools

## What It Does
Controls music playback on the BEAN robot via WebSocket commands. Maps natural language genre requests to SD card folder names.

## Tools
- `play_music`, `stop_music`, `next_track`, `set_volume`, `pause_music`, `resume_music`

## ADK Type
`LlmAgent` (Gemini 2.5 Flash) with Music `FunctionTools`

## Env Vars
- `GOOGLE_API_KEY`

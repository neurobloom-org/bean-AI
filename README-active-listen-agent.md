# Active Listen Agent — get_filler_phrase FunctionTool

## What It Does
Rule-based filler phrase selection based on detected emotion. Returns pre-cached TTS audio references for immediate playback (<50ms target).

## Inputs
- `emotion`: Current detected emotion label
- `last_phrase`: Previously used phrase (to avoid repetition)

## Outputs
- `FillerPhraseResult { phrase, audio_cache_key, audio_b64 }`

## ADK Type
`FunctionTool` — rule-based, no LLM needed.

## Env Vars
- `ELEVENLABS_API_KEY`: For pre-caching TTS audio
- `ELEVENLABS_VOICE_ID`: BEAN voice ID

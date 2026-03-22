# Bean AI — RAG System Architecture

## Overview
Bean is a therapeutic keychain companion with a multi-component AI system.
The RAG system activates only when a classifier detects therapeutic intent.
General conversations are handled by a standard LLM without RAG.
The ESP32 hardware handles physical interaction while the cloud handles all AI processing.

## Interaction Modes
Bean supports two ways to interact:

### Option 1 — Wake Word
- User says "Hey Bean..."
- ESP32 detects wake word using Porcupine (always listening, low power)
- LED lights up → Bean starts recording
- Silence detected → recording stops
- Audio sent to cloud

### Option 2 — Push to Talk
- User holds the button on Bean
- ESP32 starts recording
- User releases button → recording stops
- Audio sent to cloud

## System Flow
```
User speaks to Bean (wake word OR button)
        ↓
ESP32 records audio
        ↓
Audio sent to cloud via WiFi
        ↓
Whisper converts audio → text
        ↓
Classifier checks intent
        ↓
        ├── General talk → standard LLM response
        │
        └── Therapeutic → Crisis check runs FIRST
                        ↓
                        FAISS searches CBT/MBCT/DCT chunks
                                ↓
                        Groq/Llama3 generates response
                                ↓
                        Mood score logged to Supabase
                                ↓
                        Response sent back to ESP32
                                ↓
                        Bean speaks response aloud
                        ```

## Therapy Frameworks
| Framework | Focus | Topics |
|---|---|---|
| CBT | Thought & behaviour change | Depression, anxiety, stress |
| MBCT | Mindfulness & acceptance | Trauma, grief, self-harm |
| DCT | Compassion & relationships | Relationships, family, intimacy |

## Tech Stack
| Layer | Tool | Cost |
|---|---|---|
| Wake word | Porcupine (Picovoice) | Free tier |
| Speech to text | Whisper tiny | Free |
| Embeddings | all-MiniLM-L6-v2 | Free |
| Vector search | FAISS | Free |
| LLM | Groq — Llama 3.1 8B | Free tier |
| Database | Supabase | Free tier |
| Hosting | Railway | Free tier |
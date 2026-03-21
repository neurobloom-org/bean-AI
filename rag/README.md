# Bean — RAG System

## What is this?
This is the RAG (Retrieval-Augmented Generation) system for Bean.
It only activates when the classifier detects a therapeutic conversation.

## Folder Structure
```
rag/
├── pipeline/          # Build the FAISS index from datasets
│   ├── build_index.py
│   ├── test_pipeline.py
│   └── requirements.txt
│
├── cloud/             # FastAPI server deployed to Railway
│   ├── main.py        # API endpoints
│   ├── rag.py         # RAG engine
│   ├── crisis.py      # Crisis detection
│   ├── mood.py        # Mood scoring
│   ├── requirements.txt
│   └── .env.example
│
├── notebooks/         # Google Colab notebook
│   └── build_index.ipynb
│
├── .gitignore
└── ARCHITECTURE.md
```

## Quick Start
1. Build index → open `notebooks/build_index.ipynb` in Google Colab
2. Deploy server → see `cloud/` folder
3. Full architecture → see `ARCHITECTURE.md`

## Datasets Used
- `combined_dataset.json` — Therapist Q&A pairs
- `counsel_chat2.csv` — CounselChat with topic labels
- `counselchat-data.csv` — CounselChat extended
- `emotion-emotion_69k.csv` — Empathetic dialogues

## My Role
Built by Madusha Kolambage — AI Department, Bean Project
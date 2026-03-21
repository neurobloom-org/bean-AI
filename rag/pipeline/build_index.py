"""
Bean RAG Pipeline — build_index.py
====================================
Cleans your therapy datasets and builds the FAISS vector index.

Run this on Google Colab:
    1. Upload your dataset files
    2. Run: python build_index.py
    3. Download: faiss_index.bin + chunks.json
"""

import re
import html

# Dataset file paths
# Make sure these files are in the same folder when running

PATHS = {
    "combined_json": "combined_dataset.json",
    "counsel_chat2": "counsel_chat2.csv",
    "counselchat": "counselchat-data.csv",
    "emotion": "emotion-emotion_69k.csv",
}

# Cleaning Functions


def clean_html_text(text):
    """Remove HTML tags and fix special characters."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = re.sub(r"<[^>]+>", " ", text)  # remove HTML tags
    text = html.unescape(text)  # fix &amp; &#34; etc
    text = re.sub(r"https?://\S+", "", text)  # remove URLs
    text = re.sub(r"\s+", " ", text).strip()  # clean extra spaces
    return text


def is_quality_response(text):
    """Filter out junk responses that are too short or too long."""
    if not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < 80:  # too short to be useful
        return False
    if len(text) > 8000:  # too long - Outlier wall of text
        return False
    return True


# ─ Topic to Framework Mapping ─
# Maps CounselChat topics to CBT / MBCT / DCT frameworks

TOPIC_TO_FRAMEWORK = {
    # CBT — structured thought and behaviour change
    "depression": "cbt",
    "anxiety": "cbt",
    "behavioral-change": "cbt",
    "self-esteem": "cbt",
    "stress": "cbt",
    "anger-management": "cbt",
    "eating-disorders": "cbt",
    "substance-abuse": "cbt",
    "sleep-improvement": "cbt",
    # MBCT — mindfulness and acceptance
    "trauma": "mbct",
    "grief-and-loss": "mbct",
    "self-harm": "mbct",
    "spirituality": "mbct",
    # DCT — compassion and relationships
    "relationships": "dct",
    "intimacy": "dct",
    "family-conflict": "dct",
    "marriage": "dct",
    "parenting": "dct",
    "domestic-violence": "dct",
}

# - Chunking Function -


def make_chunk_text(context, response):
    """Format a question and answer into one chunk for RAG."""
    return f"User concern: {context.strip()}\n\nTherapist response: {response.strip()}"


def chunk_records(records):
    """Convert records into chunks ready for embedding."""
    print("Chunking records...")
    chunks = []
    for r in records:
        text = make_chunk_text(r["context"], r["response"])
        # Keep chunks under 1200 characters for best embedding quality
        if len(text) > 1200:
            text = text[:1200]
        chunks.append(
            {
                "text": text,
                "framework": r["framework"],
                "topic": r["topic"],
                "source": r["source"],
            }
        )
    print(f"  → {len(chunks)} chunks created")
    return chunks

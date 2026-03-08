import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load from root directory
load_dotenv()

# Configure the API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_768_embedding(text):
    """
    Generates a 768-dimensional vector for memory storage.
    """
    try:
        # Attempt 1: The modern model
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        # Attempt 2: The legacy stable model if 004 fails
        print(f"Notice: 004 failed, trying fallback model... ({e})")
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
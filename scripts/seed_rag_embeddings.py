import os
import json
import uuid
from typing import List
from supabase import create_client, Client

# --- CONFIGURATION ---
# Replace these with your actual Supabase URL and Service Role Key if not in environment
SUPABASE_URL = os.environ.get("SUPABASE_URL", "your-supabase-url")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "your-service-role-key")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_dummy_embedding(text: str) -> List[float]:
    """
    Placeholder for actual OpenAI Embedding. 
    In production, you would use: openai.Embedding.create(...)
    This returns a 1536-dimension vector of random small floats for testing.
    """
    import random
    return [random.uniform(-0.1, 0.1) for _ in range(1536)]

def seed_techniques():
    print("🌱 Starting RAG seeding...")
    
    # These match the techniques seen in your screenshot
    techniques = [
        {
            "category": "DBT",
            "technique_name": "4-7-8 Breathing",
            "content": "Inhale for 4 seconds, hold for 7 seconds, exhale for 8 seconds. This resets the nervous system."
        },
        {
            "category": "CBT",
            "technique_name": "Behavioral Activation",
            "content": "Schedule small, manageable activities that bring a sense of accomplishment or pleasure."
        },
        {
            "category": "CBT",
            "technique_name": "Thought Challenging",
            "content": "Identify the negative thought, look for evidence for/against it, and find a balanced perspective."
        },
        {
            "category": "General",
            "technique_name": "Active Listening",
            "content": "Bean should repeat back what it heard to ensure the user feels understood before giving advice."
        }
    ]

    for tech in techniques:
        print(f"  -> Processing: {tech['technique_name']}")
        embedding = get_dummy_embedding(tech['content'])
        
        # Upsert into the rag_techniques table
        data, count = supabase.table("rag_techniques").upsert({
            "category": tech['category'],
            "technique_name": tech['technique_name'],
            "content": tech['content'],
            "embedding": embedding
        }, on_conflict="technique_name").execute()

    print("✅ Seeding complete! Check your Supabase dashboard now.")

if __name__ == "__main__":
    if SUPABASE_URL == "your-supabase-url":
        print("❌ Error: Please set your SUPABASE_URL and SUPABASE_SERVICE_KEY first.")
    else:
        seed_techniques()
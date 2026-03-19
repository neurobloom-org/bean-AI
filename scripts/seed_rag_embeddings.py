import os
import random
from typing import List
from supabase import create_client, Client

# --- CONFIGURATION ---
SB_URL: str = os.environ.get("SUPABASE_URL", "your-supabase-url")
SB_KEY: str = os.environ.get("SUPABASE_SERVICE_KEY", "your-service-role-key")

supabase_client: Client = create_client(SB_URL, SB_KEY)


def get_dummy_embedding(text: str) -> List[float]:
    """
    Placeholder for actual OpenAI Embedding.
    Returns a 1536-dimension vector of random small floats for testing.
    """
    random.seed(len(text))
    return [random.uniform(-0.1, 0.1) for _ in range(1536)]


def seed_techniques() -> None:
    """
    Seeds the RAG techniques into the Supabase database.
    """
    print("🌱 Starting RAG seeding...")

    techniques = [
        {
            "category": "DBT",
            "technique_name": "4-7-8 Breathing",
            "content": "Inhale for 4s, hold 7s, exhale 8s.",
        },
        {
            "category": "CBT",
            "technique_name": "Behavioral Activation",
            "content": "Schedule activities for pleasure.",
        },
        {
            "category": "CBT",
            "technique_name": "Thought Challenging",
            "content": "Identify negative thoughts and find balance.",
        },
        {
            "category": "General",
            "technique_name": "Active Listening",
            "content": "Bean repeats back to ensure understanding.",
        },
    ]

    for tech in techniques:
        print(f"  -> Processing: {tech['technique_name']}")
        embedding = get_dummy_embedding(tech['content'])

        supabase_client.table("rag_techniques").upsert(
            {
                "category": tech["category"],
                "technique_name": tech["technique_name"],
                "content": tech["content"],
                "embedding": embedding,
            },
            on_conflict="technique_name",
        ).execute()

    print("✅ Seeding complete!")


if __name__ == "__main__":
    if SB_URL == "your-supabase-url":
        print("❌ Error: Set SUPABASE_URL and SUPABASE_SERVICE_KEY.")
    else:
        seed_techniques()
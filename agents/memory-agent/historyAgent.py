import sys
import os
import json

# Ensures the project root is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.embedding_utils import get_768_embedding
from utils.supabase_client import supabase

class HistoryAgent:
    def __init__(self, user_id):
        self.user_id = user_id

    def process_incoming_message(self, text, mood_score=None, emotion=None, category=None):
        """
        Routes messages to Episodic, Semantic, or Procedural based on content.
        """
        if len(text.split()) < 3:
            return "Discarded"

        vector = get_768_embedding(text)

        # 1. Procedural Routing (Coping mechanisms/How-to behaviors)
        # Triggered if the text describes a 'way' to fix a problem or a suggestion
        if "suggest" in text.lower() or "how to" in text.lower() or category == "Coping Mechanism":
            # Example action steps for the JSONB column
            steps = ["Breathe in for 4s", "Hold for 4s", "Exhale for 4s"]
            self._save_procedural(text, steps, "Coping Mechanism")

        # 2. Episodic Routing (Events/Feelings)
        elif any(word in text.lower() for word in ["feel", "went", "happened", "today"]):
            self._save_episodic(text, vector, emotion)
        
        # 3. Semantic Routing (Facts/Preferences)
        elif any(word in text.lower() for word in ["like", "prefer", "my name", "study"]):
            self._save_semantic(text, vector)
        
        # 4. Mood Logs (Always log if emotion is present)
        if emotion:
            self._save_mood_log(emotion, mood_score, text)

    def _save_procedural(self, condition, steps, category):
        """
        Stores 'how-to' behaviors in the procedural_memory table.
        Matches columns: trigger_condition, action_steps, success_rate, category.
        """
        data = {
            "trigger_condition": condition,
            "action_steps": steps, # Supabase client handles list to JSON conversion
            "success_rate": 1.0,    # Initial success rate
            "category": category
        }
        return supabase.table("procedural_memory").insert(data).execute()

    def _save_episodic(self, text, vector, emotion):
        data = {
            "user_id": self.user_id,
            "content": text,
            "embedding": vector,
            "importance": 0.8,
            "sentiment_label": emotion
        }
        return supabase.table("episodic_memory").insert(data).execute()

    def _save_semantic(self, text, vector):
        data = {
            "user_id": self.user_id,
            "fact_text": text,
            "embedding": vector,
            "access_count": 1,
            "is_verified": False
        }
        return supabase.table("semantic_memory").insert(data).execute()

    def _save_mood_log(self, emotion, score, notes):
        data = {
            "user_id": self.user_id,
            "emotion": emotion,
            "score": score,
            "source": "AI_Inference",
            "notes": notes
        }
        return supabase.table("mood_logs").insert(data).execute()

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Ensure you use a real UUID from your auth.users or public.users table
    AGENT_USER_ID = "your-user-id-here" 
    test_agent = HistoryAgent(AGENT_USER_ID)
    
    print("Testing Procedural Memory...")
    test_agent.process_incoming_message("suggest a grounding technique for level 8 anxiety", category="Coping Mechanism")
    
    print("Testing Episodic & Mood Log...")
    test_agent.process_incoming_message("I went to the park and felt peaceful", mood_score=8, emotion="Happy")
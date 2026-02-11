"""
Configuration settings for the voice therapy system
Loads environment variables and provides centralized config
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """
    Centralized configuration for all agents
    """
    
    # Google Cloud
    GOOGLE_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Audio Configuration
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))
    LANGUAGE_CODE = os.getenv("LANGUAGE_CODE", "en-US")
    VOICE_NAME = os.getenv("VOICE_NAME", "en-US-Neural2-J")
    CHUNK_DURATION_MS = int(os.getenv("CHUNK_DURATION_MS", 100))
    
    # Classification Thresholds
    CASUAL_THRESHOLD = float(os.getenv("CASUAL_THRESHOLD", 0.5))
    THERAPEUTIC_THRESHOLD = float(os.getenv("THERAPEUTIC_THRESHOLD", 0.5))
    
    # RAG Configuration
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 3))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 100))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
    
    # Model Names
    GEMINI_MODEL = "gemini-2.0-flash-exp"
    
    # System Prompts
    CASUAL_SYSTEM_PROMPT = """Talk naturally like a friend. Keep it brief. Don't force questions at the end. Go with the flow.no emojis remeber this will covert to a voice"""
    
    THERAPEUTIC_SYSTEM_PROMPT = """You're a supportive listener. Use the context provided.
Talk like a caring friend, not a therapist. Say "That sounds tough" not formal language.
If serious, suggest professional help. Don't diagnose."""
    
    CLASSIFIER_SYSTEM_PROMPT = """You are a message classifier that categorizes user input.
Analyze the message and return confidence scores for two categories:
- casual: General conversation, small talk, daily activities, questions
- therapeutic: Mental health, emotions, stress, anxiety, depression, trauma, therapy-related

Return ONLY a JSON object with decimal scores that sum to 1.0."""
    
    @classmethod
    def validate(cls):
        """
        Validate that all required settings are present
        """
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        if not os.path.exists(cls.GOOGLE_CREDENTIALS):
            raise ValueError(f"Google credentials file not found: {cls.GOOGLE_CREDENTIALS}")
        
        print("[Settings] Configuration validated successfully")
        return True


# Create global settings instance
settings = Settings()

# Validate on import
if __name__ != "__main__":
    settings.validate()
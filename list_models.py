import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load your .env from the root folder
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("Checking available embedding models...")

try:
    for m in genai.list_models():
        if "embedContent" in m.supported_generation_methods:
            print(f"✅ Found: {m.name}")
except Exception as e:
    print(f"❌ Error: {e}")

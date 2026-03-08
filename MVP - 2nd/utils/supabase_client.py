import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

# This variable NAME must be 'supabase' to match your historyAgent import
supabase: Client = create_client(url, key)
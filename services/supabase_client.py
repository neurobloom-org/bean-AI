"""
services/supabase_client.py
============================
Supabase client initialization.
Only needs shared/ — first service to build.
"""

from supabase import Client, create_client

from shared.config import config


def get_supabase_client() -> Client:
    """Create and return a Supabase client instance."""
    return create_client(config.SUPABASE_URL, config.SUPABASE_KEY)


# Single instance used across all services
supabase: Client = get_supabase_client()

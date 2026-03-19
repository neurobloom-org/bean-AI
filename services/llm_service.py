"""
services/llm_service.py
========================
LLM service using Google Gemini.
Only needs shared/config.
"""

import google.generativeai as genai

from shared.config import config

genai.configure(api_key=config.GOOGLE_API_KEY)


def get_llm_cheap():
    """Get the cheap/fast model for simple tasks."""
    return genai.GenerativeModel(config.LLM_CHEAP_MODEL)


def get_llm_pro():
    """Get the pro model for complex tasks."""
    return genai.GenerativeModel(config.LLM_PRO_MODEL)


async def generate_response(prompt: str, use_pro: bool = False) -> str:
    """Generate a response from the LLM."""
    model = get_llm_pro() if use_pro else get_llm_cheap()
    response = await model.generate_content_async(prompt)
    return response.text

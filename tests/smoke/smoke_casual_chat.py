"""Smoke test — actually runs the agent against Gemini Flash."""



import asyncio
import os
import sys

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

from agents.casual_chat.agent import casual_chat_agent, fallback_response, sanitize_response


async def run(user_text: str, emotion: str, memory: str = "No memory context yet.") -> str:
    runner = InMemoryRunner(agent=casual_chat_agent)

    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="test-user",
        state={
            "current_emotion": emotion,
            "memory_context": memory,
        },
    )

    response_text = ""
    async for event in runner.run_async(
        user_id="test-user",
        session_id=session.id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_text)],
        ),
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    response_text = part.text

    return sanitize_response(response_text, emotion) if response_text.strip() else fallback_response(emotion)

async def main():
    cases = [
        ("hey what's up",               "neutral", "No memory context yet."),
        ("I had the worst day",          "sad",     "No memory context yet."),
        ("I just got an A on my test!",  "happy",   "User profile:\nName: Isara"),
        ("I'm so angry at my friend",    "angry",   "No memory context yet."),
    ]

    for text, emotion, memory in cases:
        response = await run(text, emotion, memory)
        print(f"\n[{emotion.upper()}] User: {text}")
        print(f"BEAN: {response}")
        sentence_count = response.count("\n") + 1
        print(f"Sentences: {sentence_count}")


if __name__ == "__main__":
    asyncio.run(main())
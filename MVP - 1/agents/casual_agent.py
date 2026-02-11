"""
Casual Agent
Handles everyday casual conversations using Gemini
"""

import google.generativeai as genai
from config.settings import settings


class CasualAgent:
    """
    Agent for handling casual, everyday conversations
    Uses Gemini with a friendly conversational prompt
    """
    
    def __init__(self):
        """
        Initialize Casual agent with Gemini
        """
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            settings.GEMINI_MODEL,
            system_instruction=settings.CASUAL_SYSTEM_PROMPT
        )
        self.chat_session = None
        
        print("[Casual Agent] Initialized with Gemini")
    
    
    def generate_response(self, user_message: str) -> dict:
        """
        Generate a casual conversational response
        
        Args:
            user_message: User's message
            
        Returns:
            Dictionary with 'response' and 'success' keys
        """
        try:
            print(f"[Casual Agent] Processing: '{user_message}'")
            
            # Generate response
            response = self.model.generate_content(user_message)
            response_text = response.text.strip()
            
            print(f"[Casual Agent] Response: '{response_text}'")
            
            return {
                'response': response_text,
                'success': True,
                'agent': 'casual'
            }
        
        except Exception as e:
            print(f"[Casual Agent] Error: {e}")
            return {
                'response': "I'm sorry, I'm having trouble responding right now. Could you try again?",
                'success': False,
                'error': str(e),
                'agent': 'casual'
            }
    
    
    def start_conversation(self):
        """
        Start a new conversation session
        Maintains context across multiple messages
        """
        self.chat_session = self.model.start_chat(history=[])
        print("[Casual Agent] New conversation started")
    
    
    def continue_conversation(self, user_message: str) -> dict:
        """
        Continue an ongoing conversation with context
        
        Args:
            user_message: User's message
            
        Returns:
            Dictionary with response
        """
        try:
            if self.chat_session is None:
                self.start_conversation()
            
            print(f"[Casual Agent] Continuing conversation: '{user_message}'")
            
            response = self.chat_session.send_message(user_message)
            response_text = response.text.strip()
            
            print(f"[Casual Agent] Response: '{response_text}'")
            
            return {
                'response': response_text,
                'success': True,
                'agent': 'casual'
            }
        
        except Exception as e:
            print(f"[Casual Agent] Error: {e}")
            return {
                'response': "I'm sorry, I'm having trouble responding right now.",
                'success': False,
                'error': str(e),
                'agent': 'casual'
            }
    
    
    def reset_conversation(self):
        """
        Reset the conversation history
        """
        self.chat_session = None
        print("[Casual Agent] Conversation reset")


# Test function
if __name__ == "__main__":
    print("Testing Casual Agent...")
    agent = CasualAgent()
    
    test_messages = [
        "Hey, how are you?",
        "What's your favorite color?",
        "Tell me a fun fact",
        "What should I cook for dinner tonight?"
    ]
    
    print("\n" + "="*60)
    for msg in test_messages:
        print(f"\nUser: {msg}")
        result = agent.generate_response(msg)
        print(f"Agent: {result['response']}")
        print("-"*60)
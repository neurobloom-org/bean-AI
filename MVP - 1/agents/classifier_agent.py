"""
Classifier Agent
Classifies user input as 'casual' or 'therapeutic' using Gemini
"""

import google.generativeai as genai
import json
from config.settings import settings


class ClassifierAgent:
    """
    Agent responsible for classifying user messages
    Returns confidence scores for casual vs therapeutic conversation
    """
    
    def __init__(self):
        """
        Initialize Classifier agent with Gemini
        """
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
        self.system_prompt = settings.CLASSIFIER_SYSTEM_PROMPT
        
        print("[Classifier Agent] Initialized with Gemini")
    
    
    def classify(self, text: str) -> dict:
        """
        Classify text into casual or therapeutic categories
        
        Args:
            text: User input text
            
        Returns:
            Dictionary with 'casual' and 'therapeutic' confidence scores
        """
        try:
            print(f"[Classifier Agent] Classifying: '{text}'")
            
            # Create prompt
            prompt = f"""{self.system_prompt}

User message: "{text}"

Response (JSON only):"""
            
            # Generate classification
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean response (remove markdown code blocks if present)
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            scores = json.loads(response_text)
            
            casual_score = float(scores.get('casual', 0.0))
            therapeutic_score = float(scores.get('therapeutic', 0.0))
            
            # Normalize scores to sum to 1.0
            total = casual_score + therapeutic_score
            if total > 0:
                casual_score /= total
                therapeutic_score /= total
            
            print(f"[Classifier Agent] Casual: {casual_score:.2%}, Therapeutic: {therapeutic_score:.2%}")
            
            return {
                'casual': casual_score,
                'therapeutic': therapeutic_score,
                'success': True
            }
        
        except json.JSONDecodeError as e:
            print(f"[Classifier Agent] JSON Parse Error: {e}")
            print(f"[Classifier Agent] Raw response: {response_text}")
            
            # Fallback: default classification
            return {
                'casual': 0.5,
                'therapeutic': 0.5,
                'success': False,
                'error': 'Failed to parse classification'
            }
        
        except Exception as e:
            print(f"[Classifier Agent] Error: {e}")
            return {
                'casual': 0.5,
                'therapeutic': 0.5,
                'success': False,
                'error': str(e)
            }
    
    
    def route_message(self, text: str) -> str:
        """
        Classify and determine which agent should handle the message
        
        Args:
            text: User input text
            
        Returns:
            'casual' or 'therapeutic'
        """
        scores = self.classify(text)
        
        if scores['therapeutic'] > scores['casual']:
            route = 'therapeutic'
        else:
            route = 'casual'
        
        print(f"[Classifier Agent] Routing to: {route.upper()} agent")
        return route


# Test function
if __name__ == "__main__":
    print("Testing Classifier Agent...")
    agent = ClassifierAgent()
    
    test_messages = [
        "I'm feeling really anxious about my job interview tomorrow",
        "What's the weather like today?",
        "I can't sleep and I've been feeling depressed",
        "Do you know any good restaurants nearby?",
        "I'm struggling with panic attacks"
    ]
    
    print("\n" + "="*60)
    for msg in test_messages:
        print(f"\nMessage: '{msg}'")
        result = agent.classify(msg)
        route = agent.route_message(msg)
        print(f"Route: {route.upper()}")
        print("-"*60)
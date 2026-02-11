"""
Therapeutic Agent
Handles mental health conversations using RAG + Gemini
"""

import google.generativeai as genai
from config.settings import settings
from rag.retriever import RAGRetriever


class TherapeuticAgent:
    """
    Agent for handling therapeutic conversations
    Uses RAG to retrieve relevant mental health knowledge
    Combines with Gemini for empathetic responses
    """
    
    def __init__(self, rag_retriever: RAGRetriever = None):
        """
        Initialize Therapeutic agent with Gemini and RAG
        
        Args:
            rag_retriever: RAGRetriever instance (creates new if None)
        """
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            settings.GEMINI_MODEL,
            system_instruction=settings.THERAPEUTIC_SYSTEM_PROMPT
        )
        
        # Initialize RAG retriever
        self.rag_retriever = rag_retriever or RAGRetriever()
        
        self.chat_session = None
        
        print("[Therapeutic Agent] Initialized with Gemini + RAG")
    
    
    def generate_response(self, user_message: str, use_rag: bool = True) -> dict:
        """
        Generate a therapeutic response using RAG + LLM
        
        Args:
            user_message: User's message
            use_rag: Whether to use RAG for context retrieval
            
        Returns:
            Dictionary with 'response' and 'success' keys
        """
        try:
            print(f"[Therapeutic Agent] Processing: '{user_message}'")
            
            # Build prompt with RAG context if enabled
            if use_rag:
                context = self.rag_retriever.retrieve_context(
                    user_message, 
                    top_k=settings.TOP_K_RESULTS
                )
                
                if context:
                    full_prompt = f"""Relevant therapeutic knowledge:

{context}

User's message: {user_message}

Please provide an empathetic, supportive response based on the context above. Be warm, validating, and helpful."""
                else:
                    full_prompt = f"""User's message: {user_message}

Please provide an empathetic, supportive response. Be warm, validating, and helpful."""
            else:
                full_prompt = user_message
            
            # Generate response
            response = self.model.generate_content(full_prompt)
            response_text = response.text.strip()
            
            print(f"[Therapeutic Agent] Response generated (length: {len(response_text)} chars)")
            
            return {
                'response': response_text,
                'success': True,
                'agent': 'therapeutic',
                'used_rag': use_rag and bool(context)
            }
        
        except Exception as e:
            print(f"[Therapeutic Agent] Error: {e}")
            return {
                'response': "I'm here to support you. Would you like to tell me more about what you're experiencing?",
                'success': False,
                'error': str(e),
                'agent': 'therapeutic',
                'used_rag': False
            }
    
    
    def start_conversation(self):
        """
        Start a new therapeutic conversation session
        """
        self.chat_session = self.model.start_chat(history=[])
        print("[Therapeutic Agent] New conversation started")
    
    
    def continue_conversation(self, user_message: str, use_rag: bool = True) -> dict:
        """
        Continue therapeutic conversation with context
        
        Args:
            user_message: User's message
            use_rag: Whether to use RAG
            
        Returns:
            Dictionary with response
        """
        try:
            if self.chat_session is None:
                self.start_conversation()
            
            print(f"[Therapeutic Agent] Continuing conversation: '{user_message}'")
            
            # Get RAG context if enabled
            context = ""
            if use_rag:
                context = self.rag_retriever.retrieve_context(
                    user_message,
                    top_k=settings.TOP_K_RESULTS
                )
            
            # Build message with context
            if context:
                message = f"""Relevant knowledge:
{context}

User says: {user_message}"""
            else:
                message = user_message
            
            # Send message
            response = self.chat_session.send_message(message)
            response_text = response.text.strip()
            
            print(f"[Therapeutic Agent] Response generated")
            
            return {
                'response': response_text,
                'success': True,
                'agent': 'therapeutic',
                'used_rag': use_rag and bool(context)
            }
        
        except Exception as e:
            print(f"[Therapeutic Agent] Error: {e}")
            return {
                'response': "I'm here to listen and support you. Please continue.",
                'success': False,
                'error': str(e),
                'agent': 'therapeutic',
                'used_rag': False
            }
    
    
    def reset_conversation(self):
        """
        Reset the conversation history
        """
        self.chat_session = None
        print("[Therapeutic Agent] Conversation reset")
    
    
    def add_knowledge(self, documents: list):
        """
        Add new therapeutic knowledge to the RAG system
        
        Args:
            documents: List of text documents to add
        """
        self.rag_retriever.vector_store.add_documents(documents)
        print(f"[Therapeutic Agent] Added {len(documents)} documents to knowledge base")


# Test function
if __name__ == "__main__":
    print("Testing Therapeutic Agent...")
    
    # Initialize with sample knowledge
    from rag.vector_store import VectorStore
    from rag.retriever import RAGRetriever
    
    store = VectorStore()
    sample_docs = [
        "When experiencing anxiety, try the 5-4-3-2-1 grounding technique: identify 5 things you see, 4 you can touch, 3 you hear, 2 you smell, and 1 you taste.",
        "It's okay to not be okay. Seeking help is a sign of strength, not weakness. Professional therapists can provide valuable support.",
        "Depression is treatable. Common treatments include therapy (especially CBT), medication, exercise, and social support. Recovery is possible."
    ]
    store.add_documents(sample_docs)
    
    retriever = RAGRetriever(vector_store=store)
    agent = TherapeuticAgent(rag_retriever=retriever)
    
    test_messages = [
        "I'm having a panic attack and I don't know what to do",
        "I feel so alone and sad all the time",
        "Is it normal to feel this anxious?"
    ]
    
    print("\n" + "="*60)
    for msg in test_messages:
        print(f"\nUser: {msg}")
        result = agent.generate_response(msg)
        print(f"\nAgent: {result['response']}")
        print(f"Used RAG: {result.get('used_rag', False)}")
        print("="*60)
"""
Main Orchestrator for Voice Therapy System
Coordinates all agents: STT → Classifier → Casual/Therapeutic → TTS
"""

import sounddevice as sd
import time
from config.settings import settings
from utils.audio_handler import AudioHandler
from agents.speech_to_text_agent import SpeechToTextAgent
from agents.classifier_agent import ClassifierAgent
from agents.casual_agent import CasualAgent
from agents.therapeutic_agent import TherapeuticAgent
from agents.text_to_speech_agent import TextToSpeechAgent
from rag.vector_store import VectorStore
from rag.retriever import RAGRetriever


class VoiceTherapySystem:
    """
    Main orchestrator for the multi-agent voice therapy system
    Manages the flow: Voice Input → STT → Classifier → Agent → TTS → Voice Output
    """
    
    def __init__(self):
        """
        Initialize all agents and components
        """
        print("="*70)
        print("INITIALIZING VOICE THERAPY SYSTEM")
        print("="*70)
        
        # Initialize audio handler
        self.audio_handler = AudioHandler(sample_rate=settings.SAMPLE_RATE)
        
        # Initialize agents
        print("\n[System] Initializing agents...")
        self.stt_agent = SpeechToTextAgent()
        self.classifier_agent = ClassifierAgent()
        self.casual_agent = CasualAgent()
        
        # Initialize RAG system for therapeutic agent
        print("\n[System] Initializing RAG system...")
        self.vector_store = VectorStore()
        self.rag_retriever = RAGRetriever(vector_store=self.vector_store)
        self.therapeutic_agent = TherapeuticAgent(rag_retriever=self.rag_retriever)
        
        self.tts_agent = TextToSpeechAgent()
        
        # Load knowledge base if available
        self._load_knowledge_base()
        
        # System state
        self.is_running = False
        self.conversation_count = 0
        
        print("\n" + "="*70)
        print("SYSTEM READY!")
        print("="*70)
    
    
    def _load_knowledge_base(self):
        """
        Load therapeutic knowledge from files into RAG system
        """
        import os
        knowledge_dir = "./data/therapy_knowledge"
        
        if os.path.exists(knowledge_dir):
            print(f"[System] Loading knowledge from: {knowledge_dir}")
            
            for filename in os.listdir(knowledge_dir):
                if filename.endswith('.txt'):
                    filepath = os.path.join(knowledge_dir, filename)
                    try:
                        self.vector_store.add_documents_from_file(filepath)
                    except Exception as e:
                        print(f"[System] Error loading {filename}: {e}")
            
            stats = self.vector_store.get_collection_stats()
            print(f"[System] Knowledge base loaded: {stats['document_count']} documents")
        else:
            print(f"[System] No knowledge base found at {knowledge_dir}")
            print(f"[System] Creating directory for future use...")
            os.makedirs(knowledge_dir, exist_ok=True)
    
    
    def process_user_input(self, audio_data):
        """
        Process complete pipeline for one user input
        
        Args:
            audio_data: Recorded audio as numpy array
            
        Returns:
            Dictionary with processing results
        """
        print("\n" + "="*70)
        print(f"PROCESSING INPUT #{self.conversation_count + 1}")
        print("="*70)
        
        # Step 1: Speech-to-Text
        print("\n[STEP 1] Speech-to-Text")
        print("-"*70)
        stt_result = self.stt_agent.transcribe_audio(audio_data)
        
        if not stt_result['success']:
            print(f"[System] STT failed: {stt_result.get('error', 'Unknown error')}")
            return {'success': False, 'error': 'Speech recognition failed'}
        
        user_text = stt_result['text']
        print(f"User said: \"{user_text}\"")
        
        # Step 2: Classification
        print("\n[STEP 2] Classification")
        print("-"*70)
        classification = self.classifier_agent.classify(user_text)
        
        casual_score = classification['casual']
        therapeutic_score = classification['therapeutic']
        
        print(f"Casual: {casual_score:.1%} | Therapeutic: {therapeutic_score:.1%}")
        
        # Step 3: Route to appropriate agent
        print("\n[STEP 3] Response Generation")
        print("-"*70)
        
        if therapeutic_score > casual_score:
            print("Routing to: THERAPEUTIC AGENT (with RAG)")
            agent_response = self.therapeutic_agent.generate_response(user_text, use_rag=True)
        else:
            print("Routing to: CASUAL AGENT")
            agent_response = self.casual_agent.generate_response(user_text)
        
        if not agent_response['success']:
            print(f"[System] Response generation failed")
            return {'success': False, 'error': 'Response generation failed'}
        
        response_text = agent_response['response']
        print(f"\nAgent response: \"{response_text}\"")
        
        if agent_response.get('used_rag'):
            print("[Info] Response used RAG context")
        
        # Step 4: Text-to-Speech
        print("\n[STEP 4] Text-to-Speech")
        print("-"*70)
        tts_result = self.tts_agent.synthesize_speech(response_text)
        
        if not tts_result['success']:
            print(f"[System] TTS failed: {tts_result.get('error', 'Unknown error')}")
            return {'success': False, 'error': 'Speech synthesis failed'}
        
        # Step 5: Play audio response
        print("\n[STEP 5] Playing Audio Response")
        print("-"*70)
        self.audio_handler.play_audio(
            tts_result['audio_data'],
            sample_rate=tts_result['sample_rate']
        )
        
        print("\n" + "="*70)
        print("PROCESSING COMPLETE")
        print("="*70)
        
        self.conversation_count += 1
        
        return {
            'success': True,
            'user_text': user_text,
            'classification': classification,
            'response_text': response_text,
            'agent_type': agent_response['agent']
        }
    
    
    def run_single_interaction(self, recording_duration: float = 5.0):
        """
        Run a single voice interaction
        
        Args:
            recording_duration: How long to record user input (seconds)
        """
        print("\n" + "="*70)
        print("STARTING NEW INTERACTION")
        print("="*70)
        print(f"\nListening for {recording_duration} seconds...")
        print("Speak now!")
        
        # Record audio
        audio_data = self.audio_handler.record_audio(duration=recording_duration)
        
        # Process through pipeline
        result = self.process_user_input(audio_data)
        
        return result
    
    
    def run_continuous_mode(self, recording_duration: float = 5.0):
        """
        Run in continuous conversation mode
        
        Args:
            recording_duration: Duration for each recording
        """
        self.is_running = True
        
        print("\n" + "="*70)
        print("CONTINUOUS MODE ACTIVATED")
        print("="*70)
        print("\nPress Ctrl+C to stop")
        print(f"Each recording will be {recording_duration} seconds")
        
        try:
            while self.is_running:
                self.run_single_interaction(recording_duration)
                
                # Brief pause between interactions
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n[System] Stopping continuous mode...")
            self.is_running = False
    
    
    def run_interactive_menu(self):
        """
        Run interactive menu for user to choose actions
        """
        while True:
            print("\n" + "="*70)
            print("VOICE THERAPY SYSTEM - MENU")
            print("="*70)
            print("\n1. Single Interaction (5 seconds)")
            print("2. Single Interaction (10 seconds)")
            print("3. Continuous Mode")
            print("4. Test STT Only")
            print("5. Test TTS Only")
            print("6. View System Stats")
            print("7. Exit")
            
            choice = input("\nEnter choice (1-7): ").strip()
            
            if choice == '1':
                self.run_single_interaction(recording_duration=5.0)
            
            elif choice == '2':
                self.run_single_interaction(recording_duration=10.0)
            
            elif choice == '3':
                duration = input("Recording duration per interaction (default 5s): ").strip()
                duration = float(duration) if duration else 5.0
                self.run_continuous_mode(recording_duration=duration)
            
            elif choice == '4':
                print("\nTesting STT - Speak for 5 seconds...")
                audio = self.audio_handler.record_audio(5.0)
                result = self.stt_agent.transcribe_audio(audio)
                print(f"\nTranscript: {result.get('text', 'N/A')}")
                print(f"Confidence: {result.get('confidence', 0):.2%}")
            
            elif choice == '5':
                test_text = input("\nEnter text to speak: ").strip()
                if test_text:
                    result = self.tts_agent.synthesize_speech(test_text)
                    if result['success']:
                        self.audio_handler.play_audio(result['audio_data'], result['sample_rate'])
            
            elif choice == '6':
                self._show_stats()
            
            elif choice == '7':
                print("\nExiting system. Goodbye!")
                break
            
            else:
                print("\nInvalid choice. Please try again.")
    
    
    def _show_stats(self):
        """
        Display system statistics
        """
        print("\n" + "="*70)
        print("SYSTEM STATISTICS")
        print("="*70)
        
        stats = self.vector_store.get_collection_stats()
        
        print(f"\nConversations processed: {self.conversation_count}")
        print(f"Knowledge base documents: {stats['document_count']}")
        print(f"Embedding model: {stats['embedding_model']}")
        print(f"Sample rate: {settings.SAMPLE_RATE} Hz")
        print(f"Language: {settings.LANGUAGE_CODE}")
        print(f"Voice: {settings.VOICE_NAME}")


def main():
    """
    Main entry point
    """
    print("\n")
    
    print("            VOICE THERAPY SYSTEM - MULTI-AGENT AI                   ")
    
    print("\n")
    
    try:
        # Initialize system
        system = VoiceTherapySystem()
        
        # Run interactive menu
        system.run_interactive_menu()
        
    except KeyboardInterrupt:
        print("\n\n[System] Interrupted by user")
    
    except Exception as e:
        print(f"\n[System] Fatal error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n[System] Shutdown complete")


if __name__ == "__main__":
    main()

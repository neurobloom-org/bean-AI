"""
Speech-to-Text Agent
Converts user voice input to text using Google Cloud Speech-to-Text API
"""

import io
import numpy as np
from google.cloud import speech
from config.settings import settings


class SpeechToTextAgent:
    """
    Agent responsible for converting speech to text
    Uses Google Cloud Speech-to-Text API
    """
    
    def __init__(self):
        """
        Initialize Speech-to-Text agent
        """
        self.client = speech.SpeechClient()
        self.sample_rate = settings.SAMPLE_RATE
        self.language_code = settings.LANGUAGE_CODE
        
        print(f"[STT Agent] Initialized (language={self.language_code})")
    
    
    def transcribe_audio(self, audio_data: np.ndarray) -> dict:
        """
        Transcribe audio data to text
        
        Args:
            audio_data: Audio data as numpy array (int16)
            
        Returns:
            Dictionary with 'text' and 'confidence' keys
        """
        try:
            # Convert numpy array to bytes
            audio_bytes = audio_data.tobytes()
            
            # Create recognition audio
            audio = speech.RecognitionAudio(content=audio_bytes)
            
            # Configure recognition settings
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                language_code=self.language_code,
                enable_automatic_punctuation=True,
                model="default",
                use_enhanced=True
            )
            
            print("[STT Agent] Transcribing audio...")
            
            # Perform recognition
            response = self.client.recognize(config=config, audio=audio)
            
            # Extract results
            if response.results:
                result = response.results[0]
                transcript = result.alternatives[0].transcript
                confidence = result.alternatives[0].confidence
                
                print(f"[STT Agent] Transcription: '{transcript}'")
                print(f"[STT Agent] Confidence: {confidence:.2%}")
                
                return {
                    'text': transcript,
                    'confidence': confidence,
                    'success': True
                }
            else:
                print("[STT Agent] No speech detected")
                return {
                    'text': '',
                    'confidence': 0.0,
                    'success': False,
                    'error': 'No speech detected'
                }
        
        except Exception as e:
            print(f"[STT Agent] Error: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'success': False,
                'error': str(e)
            }
    
    
    def transcribe_streaming(self, audio_generator):
        """
        Transcribe streaming audio (for future implementation)
        
        Args:
            audio_generator: Generator yielding audio chunks
            
        Yields:
            Transcription results
        """
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language_code,
            enable_automatic_punctuation=True
        )
        
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=False
        )
        
        requests = (
            speech.StreamingRecognizeRequest(audio_content=chunk)
            for chunk in audio_generator
        )
        
        responses = self.client.streaming_recognize(streaming_config, requests)
        
        for response in responses:
            for result in response.results:
                if result.is_final:
                    transcript = result.alternatives[0].transcript
                    confidence = result.alternatives[0].confidence
                    
                    yield {
                        'text': transcript,
                        'confidence': confidence,
                        'success': True
                    }


# Test function
if __name__ == "__main__":
    import sounddevice as sd
    
    print("Testing Speech-to-Text Agent...")
    agent = SpeechToTextAgent()
    
    print("\nSpeak now for 5 seconds...")
    audio = sd.rec(
        int(5 * settings.SAMPLE_RATE),
        samplerate=settings.SAMPLE_RATE,
        channels=1,
        dtype='int16'
    )
    sd.wait()
    
    print("\nTranscribing...")
    result = agent.transcribe_audio(audio)
    
    if result['success']:
        print(f"\nTranscript: {result['text']}")
        print(f"Confidence: {result['confidence']:.2%}")
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")
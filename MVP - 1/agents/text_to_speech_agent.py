"""
Text-to-Speech Agent
Converts text responses to speech using Google Cloud Text-to-Speech API
"""

import numpy as np
from google.cloud import texttospeech
from config.settings import settings


class TextToSpeechAgent:
    """
    Agent responsible for converting text to speech
    Uses Google Cloud Text-to-Speech API
    """
    
    def __init__(self):
        """
        Initialize Text-to-Speech agent
        """
        self.client = texttospeech.TextToSpeechClient()
        self.language_code = settings.LANGUAGE_CODE
        self.voice_name = settings.VOICE_NAME
        self.sample_rate = settings.SAMPLE_RATE
        
        # Configure voice - FIXED: Use FEMALE instead of NEUTRAL
        self.voice = texttospeech.VoiceSelectionParams(
            language_code=self.language_code,
            name=self.voice_name,
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        
        # Configure audio
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate
        )
        
        print(f"[TTS Agent] Initialized (voice={self.voice_name})")
    
    
    def synthesize_speech(self, text: str) -> dict:
        """
        Convert text to speech audio
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Dictionary with 'audio_data' (numpy array) and 'success' keys
        """
        try:
            print(f"[TTS Agent] Synthesizing: '{text[:50]}...'")
            
            # Create synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Perform text-to-speech request
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=self.voice,
                audio_config=self.audio_config
            )
            
            # Convert bytes to numpy array
            audio_data = np.frombuffer(response.audio_content, dtype=np.int16)
            
            print(f"[TTS Agent] Synthesis complete (audio length: {len(audio_data)} samples)")
            
            return {
                'audio_data': audio_data,
                'sample_rate': self.sample_rate,
                'success': True
            }
        
        except Exception as e:
            print(f"[TTS Agent] Error: {e}")
            return {
                'audio_data': None,
                'sample_rate': self.sample_rate,
                'success': False,
                'error': str(e)
            }
    
    
    def synthesize_ssml(self, ssml_text: str) -> dict:
        """
        Convert SSML text to speech (for advanced control)
        
        Args:
            ssml_text: SSML formatted text
            
        Returns:
            Dictionary with audio data
        """
        try:
            print(f"[TTS Agent] Synthesizing SSML...")
            
            # Create synthesis input with SSML
            synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)
            
            # Perform synthesis
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=self.voice,
                audio_config=self.audio_config
            )
            
            # Convert to numpy array
            audio_data = np.frombuffer(response.audio_content, dtype=np.int16)
            
            print(f"[TTS Agent] SSML synthesis complete")
            
            return {
                'audio_data': audio_data,
                'sample_rate': self.sample_rate,
                'success': True
            }
        
        except Exception as e:
            print(f"[TTS Agent] Error: {e}")
            return {
                'audio_data': None,
                'sample_rate': self.sample_rate,
                'success': False,
                'error': str(e)
            }
    
    
    def save_audio(self, text: str, output_file: str) -> bool:
        """
        Synthesize speech and save to file
        
        Args:
            text: Text to convert
            output_file: Output file path (.wav)
            
        Returns:
            True if successful
        """
        try:
            result = self.synthesize_speech(text)
            
            if result['success']:
                import soundfile as sf
                sf.write(
                    output_file,
                    result['audio_data'],
                    result['sample_rate']
                )
                print(f"[TTS Agent] Audio saved to: {output_file}")
                return True
            else:
                return False
        
        except Exception as e:
            print(f"[TTS Agent] Error saving audio: {e}")
            return False
    
    
    def list_available_voices(self):
        """
        List all available voices for the configured language
        Useful for testing different voices
        """
        try:
            # Get list of voices
            voices = self.client.list_voices()
            
            print(f"\n[TTS Agent] Available voices for {self.language_code}:")
            print("-" * 60)
            
            for voice in voices.voices:
                if voice.language_codes[0].startswith(self.language_code.split('-')[0]):
                    print(f"Name: {voice.name}")
                    print(f"  Gender: {texttospeech.SsmlVoiceGender(voice.ssml_gender).name}")
                    print(f"  Languages: {', '.join(voice.language_codes)}")
                    print()
        
        except Exception as e:
            print(f"[TTS Agent] Error listing voices: {e}")


# Test function
if __name__ == "__main__":
    import sounddevice as sd
    
    print("Testing Text-to-Speech Agent...")
    agent = TextToSpeechAgent()
    
    test_texts = [
        "Hello! I'm your therapeutic assistant. How are you feeling today?",
        "It's completely normal to feel anxious sometimes. Let's work through this together.",
        "I'm here to listen and support you."
    ]
    
    print("\n" + "="*60)
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: '{text}'")
        
        # Synthesize
        result = agent.synthesize_speech(text)
        
        if result['success']:
            # Play audio
            print("Playing audio...")
            sd.play(result['audio_data'], result['sample_rate'])
            sd.wait()
            print("Playback complete")
        else:
            print(f"Synthesis failed: {result.get('error', 'Unknown error')}")
        
        print("-"*60)
    
    # Optional: List available voices
    print("\nListing available voices...")
    agent.list_available_voices()
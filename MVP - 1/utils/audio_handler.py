"""
Audio handling utilities for recording and playback
Uses sounddevice for cross-platform audio support
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import queue
import threading
from typing import Optional, Callable


class AudioHandler:
    """
    Handles audio recording and playback operations
    """
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Initialize audio handler
        
        Args:
            sample_rate: Sample rate in Hz
            channels: Number of audio channels (1 for mono, 2 for stereo)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        print(f"[AudioHandler] Initialized (sample_rate={sample_rate}Hz, channels={channels})")
    
    
    def record_audio(self, duration: float) -> np.ndarray:
        """
        Record audio for a specified duration
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Recorded audio as numpy array
        """
        print(f"[AudioHandler] Recording for {duration} seconds...")
        
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='int16'
        )
        sd.wait()
        
        print("[AudioHandler] Recording complete")
        return recording
    
    
    def start_streaming_recording(self, callback: Optional[Callable] = None):
        """
        Start streaming audio recording
        Audio chunks are put into queue for processing
        
        Args:
            callback: Optional callback function for each audio chunk
        """
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"[AudioHandler] Status: {status}")
            
            if self.is_recording:
                audio_chunk = indata.copy()
                self.audio_queue.put(audio_chunk)
                
                if callback:
                    callback(audio_chunk)
        
        self.is_recording = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='int16',
            callback=audio_callback
        )
        self.stream.start()
        print("[AudioHandler] Streaming recording started")
    
    
    def stop_streaming_recording(self):
        """
        Stop streaming audio recording
        """
        if self.is_recording:
            self.is_recording = False
            self.stream.stop()
            self.stream.close()
            print("[AudioHandler] Streaming recording stopped")
    
    
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get audio chunk from queue
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Audio chunk or None if queue is empty
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    
    def play_audio(self, audio_data: np.ndarray, sample_rate: Optional[int] = None):
        """
        Play audio data
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate (uses default if None)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        print("[AudioHandler] Playing audio...")
        sd.play(audio_data, sample_rate)
        sd.wait()
        print("[AudioHandler] Playback complete")
    
    
    def save_audio(self, audio_data: np.ndarray, filename: str):
        """
        Save audio to file
        
        Args:
            audio_data: Audio data as numpy array
            filename: Output filename
        """
        sf.write(filename, audio_data, self.sample_rate)
        print(f"[AudioHandler] Audio saved to {filename}")
    
    
    def load_audio(self, filename: str) -> tuple:
        """
        Load audio from file
        
        Args:
            filename: Input filename
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio_data, sample_rate = sf.read(filename)
        print(f"[AudioHandler] Audio loaded from {filename}")
        return audio_data, sample_rate


# Test function
if __name__ == "__main__":
    print("Testing AudioHandler...")
    handler = AudioHandler()
    
    # Test recording
    print("\nRecording 3 seconds of audio...")
    audio = handler.record_audio(3.0)
    
    # Test playback
    print("\nPlaying back recorded audio...")
    handler.play_audio(audio)
    
    print("\nAudioHandler test complete!")
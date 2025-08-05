import os
import tempfile
import ollama
import wave
from piper import PiperVoice
import whisper
import sounddevice as sd
import numpy as np
import onnxruntime
import threading
import time
import queue
from scipy.signal import butter, filtfilt
from collections import deque
import yaml

import audio_ctrl

# Load configuration
def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), 'wake_word_config.yaml')
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Warning: Could not load config file: {e}")
        return {}

config = load_config()


# TODO: add voice model and whisper model to config
# voice = PiperVoice.load("voices/en_GB-southern_english_female-low.onnx", "voices/en_GB-southern_english_female-low.json")
voice = PiperVoice.load("voices/glados.onnx", "voices/glados.json")
whisper_model = whisper.load_model("base")

# Wake word detection configuration (with config file overrides)
WAKE_WORDS = config.get('wake_words', ["minion", "hey minion", "hello minion"])
AUDIO_SAMPLE_RATE = config.get('audio', {}).get('sample_rate', 16000)
CHUNK_SIZE = config.get('audio', {}).get('chunk_size', 1024)
WAKE_WORD_TIMEOUT = 10  # seconds
ENERGY_THRESHOLD = config.get('audio', {}).get('energy_threshold', 500)
SILENCE_THRESHOLD = config.get('audio', {}).get('silence_threshold', 2.0)  # Updated default
MAX_RECORDING_DURATION = config.get('audio', {}).get('max_recording_duration', 15)  # Updated default
CALIBRATION_DURATION = config.get('audio', {}).get('calibration_duration', 3)

# Assistant behavior configuration
ENABLE_ACKNOWLEDGMENT_SOUND = config.get('assistant', {}).get('enable_acknowledgment_sound', True)
ACKNOWLEDGMENT_SOUND = config.get('assistant', {}).get('acknowledgment_sound', "connected/connected.mp3")
ERROR_RESPONSE = config.get('assistant', {}).get('error_response', "Sorry, I didn't understand that command.")

# Detection configuration
ENABLE_SIMILAR_WORDS = config.get('detection', {}).get('enable_similar_words', True)
SIMILAR_WORDS = config.get('detection', {}).get('similar_words', {
    "minion": ["million", "mini on", "min eon"],
    "hey minion": ["a minion", "hay minion", "hey million"]
})

# LLM configuration
LLM_MODEL = config.get('llm', {}).get('model', 'gemma2:2b')
LLM_BASE_URL = config.get('llm', {}).get('base_url', 'http://localhost:11434')
LLM_TIMEOUT = config.get('llm', {}).get('timeout', 30)

class WakeWordDetector:
    def __init__(self):
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.stop_listening = threading.Event()
        self.energy_buffer = deque(maxlen=50)  # Rolling buffer for energy calculation
        self.calibrated_threshold = None
        self.is_speaking = False  # Flag to track when TTS is playing
        
    def calculate_audio_energy(self, audio_data):
        """Calculate RMS energy of audio data"""
        return np.sqrt(np.mean(audio_data**2))
    
    def preprocess_audio(self, audio_data):
        """Apply basic audio preprocessing"""
        # Simple high-pass filter to remove low frequency noise
        nyquist = AUDIO_SAMPLE_RATE * 0.5
        low = 300 / nyquist
        high = 3400 / nyquist
        b, a = butter(1, [low, high], btype='band')
        return filtfilt(b, a, audio_data)
    
    def calibrate_environment(self, duration=None):
        """Calibrate background noise level with smart thresholding"""
        if duration is None:
            duration = CALIBRATION_DURATION
        
        print("\nCalibrating background noise... Please stay quiet.")
        audio = sd.rec(int(duration * AUDIO_SAMPLE_RATE), 
                      samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype=np.float32)
        sd.wait()
        
        background_energy = self.calculate_audio_energy(audio.flatten())
        
        # Much more conservative thresholding based on your actual voice levels
        # TODO: ADJUST THESE!
        if background_energy > 0.1:
            # High background noise - use very low threshold
            self.calibrated_threshold = 0.05
            print(f"   High background noise ({background_energy:.3f}), using low threshold for voice detection")
        elif background_energy > 0.05:
            # Medium background noise
            self.calibrated_threshold = 0.03
            print(f"   Medium background noise ({background_energy:.3f}), using moderate threshold")
        else:
            # Low background noise
            self.calibrated_threshold = 0.007
            print(f"   Low background noise ({background_energy:.3f}), using sensitive threshold")

        print(f"   Background: {background_energy:.3f}, Threshold: {self.calibrated_threshold:.3f}")

        # Test with current audio to see if threshold is reasonable
        print("   Testing threshold with current audio levels...")
        test_audio = sd.rec(int(1 * AUDIO_SAMPLE_RATE),
                           samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype=np.float32)
        sd.wait()
        test_energy = self.calculate_audio_energy(test_audio.flatten())
        print(f"   Current audio level: {test_energy:.3f}")

        if test_energy < self.calibrated_threshold:
            print(f"   Current audio level ({test_energy:.3f}) is below threshold - good!")
        else:
            print(f"   Current audio level ({test_energy:.3f}) is above threshold ({self.calibrated_threshold:.3f})")
            # Auto-adjust if needed
            if test_energy > 0.001:  # Not complete silence
                self.calibrated_threshold = test_energy * 0.8  # Slightly below current level
                print(f"   🔧 Auto-adjusted threshold to {self.calibrated_threshold:.3f}")

        # Additional debug info
        if self.calibrated_threshold > 0.1:
            print("   ⚠️ Threshold is high - you may need to speak loudly")
        elif self.calibrated_threshold < 0.03:
            print("   ✅ Threshold is low - sensitive to quiet sounds")

    def detect_voice_activity(self, audio_chunk):
        """Detect if audio chunk contains voice activity"""
        energy = self.calculate_audio_energy(audio_chunk)
        self.energy_buffer.append(energy)
        
        # Use calibrated threshold if available, otherwise use default normalized threshold
        if self.calibrated_threshold is not None:
            threshold = self.calibrated_threshold
        else:
            # Convert ENERGY_THRESHOLD (which is for int16) to float32 scale
            threshold = ENERGY_THRESHOLD / 32767.0  # Normalize to 0-1 range
        
        # Debug: Print energy levels less frequently and only when needed
        # if len(self.energy_buffer) % 50 == 0:  # Much less frequent - every 50 chunks
        #     print(f"🔊 Energy: {energy:.3f}, Threshold: {threshold:.3f}")
        
        # Check for voice activity
        voice_detected = energy > threshold
        
        # Also consider recent energy trend for better detection
        if len(self.energy_buffer) > 3:
            recent_avg = np.mean(list(self.energy_buffer)[-3:])
            trend_detected = recent_avg > threshold * 0.8
            voice_detected = voice_detected or trend_detected
        
        # Debug: Show when voice is detected (but limit frequency)
        if voice_detected and len(self.energy_buffer) % 10 == 0:  # Only show occasionally when voice detected
            print(f"🗣️  VOICE DETECTED! Energy: {energy:.3f} > Threshold: {threshold:.3f}")
        
        return voice_detected
    
    def listen_for_wake_word(self):
        """Listen for wake word with timeout and better error handling"""        
        def audio_callback(indata, frames, time, status):
            """Callback function for audio stream"""
            if status:
                print(f"Audio status: {status}")
            try:
                self.audio_queue.put(indata.copy(), timeout=1.0)
            except queue.Full:
                # Skip if queue is full to prevent blocking
                pass
        
        try:
            with sd.InputStream(callback=audio_callback,
                              channels=1,
                              samplerate=AUDIO_SAMPLE_RATE,
                              blocksize=CHUNK_SIZE,
                              dtype=np.float32):
                
                audio_buffer = []
                last_activity_time = time.time()
                recording_triggered = False
                start_time = time.time()
                
                while not self.stop_listening.is_set():
                    try:
                        # Skip listening if TTS is currently playing
                        if self.is_speaking:
                            time.sleep(0.1)
                            continue
                        
                        # Add timeout to prevent infinite waiting
                        if time.time() - start_time > 10:  # 10 second timeout per listening session
                            print("⏰ Listening timeout, resetting...")
                            break
                        
                        # Get audio chunk from queue with timeout
                        audio_chunk = self.audio_queue.get(timeout=0.5)
                        audio_chunk = audio_chunk.flatten()
                        
                        # Preprocess audio
                        processed_chunk = self.preprocess_audio(audio_chunk)
                        
                        # Check for voice activity
                        has_voice = self.detect_voice_activity(processed_chunk)
                        
                        if has_voice:
                            last_activity_time = time.time()
                            recording_triggered = True
                            audio_buffer.extend(processed_chunk)
                            # print(f"🎙️  Voice activity detected, buffer size: {len(audio_buffer)/AUDIO_SAMPLE_RATE:.1f}s")
                            
                            # Limit buffer size (max 5 seconds for responsiveness)
                            max_samples = AUDIO_SAMPLE_RATE * 5
                            if len(audio_buffer) > max_samples:
                                audio_buffer = audio_buffer[-max_samples:]
                        
                        # Check if we should process accumulated audio
                        silence_duration = time.time() - last_activity_time
                        
                        if recording_triggered and silence_duration > SILENCE_THRESHOLD:
                            if len(audio_buffer) > AUDIO_SAMPLE_RATE * 0.5:  # At least 0.5 seconds
                                print(f"🔍 Processing {len(audio_buffer)/AUDIO_SAMPLE_RATE:.1f}s of audio...")
                                wake_word_detected = self.process_audio_for_wake_word(np.array(audio_buffer))
                                if wake_word_detected:
                                    # print("🎉 Wake word detected!")
                                    if ENABLE_ACKNOWLEDGMENT_SOUND:
                                        try:
                                            print("  Playing acknowledgment sound...")
                                            audio_ctrl.play_file(ACKNOWLEDGMENT_SOUND)
                                            # Wait a bit for the sound to finish
                                            time.sleep(0.5)  # Adjust based on your sound file length
                                        except:
                                            print("🔔 *acknowledgment sound*")
                                    return True
                            
                            # Reset for next detection
                            audio_buffer = []
                            recording_triggered = False
                            
                    except queue.Empty:
                        # No audio data available, continue
                        continue
                    except Exception as e:
                        print(f"Error in wake word detection: {e}")
                        # Clear the queue to prevent buildup
                        while not self.audio_queue.empty():
                            try:
                                self.audio_queue.get_nowait()
                            except queue.Empty:
                                break
                        continue
                        
        except Exception as e:
            print(f"Error setting up audio stream: {e}")
            return False
        
        return False
    
    def process_audio_for_wake_word(self, audio_data):
        """Process audio buffer to detect wake word"""
        try:
            # Save audio to temporary file for whisper processing
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
                # Convert to int16 for wav file
                audio_int16 = (audio_data * 32767).astype(np.int16)
                
                with wave.open(tmpfile.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(AUDIO_SAMPLE_RATE)
                    wf.writeframes(audio_int16.tobytes())
                
                # Transcribe audio
                transcribed_text = transcribe_audio(tmpfile.name)
                print(f"👂 Heard: '{transcribed_text}'")
                
                # Clean up temp file
                os.unlink(tmpfile.name)
                
                # Check for wake words
                return detect_wake_word(transcribed_text)
                
        except Exception as e:
            print(f"Error processing audio for wake word: {e}")
            return False

# Global wake word detector instance
wake_detector = WakeWordDetector()

def record_audio(filename, duration=5, fs=16000):
    print("---- RECORDING STARTED ----")
    audio = sd.rec(
        int(duration * fs), samplerate=fs, channels=1, dtype=int16
    )
    sd.wait()
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())
    print("Completed Recording...")
    
def transcribe_audio(filename):
    return whisper_model.transcribe(filename, language='en')['text']

def ask_llm(query):
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {'role': 'system', 'content': "Your name is Jetson, an autonomous rover that can interact with the world by accepting commands from the user. Only output pure raw text. Never use emojis or markdown formatting."},
            {'role': 'user', 'content': query}
        ],
    )
    return response['message']['content']

def text_to_speech(text):
    """Text to speech with speaking state management"""
    try:
        wake_detector.is_speaking = True
        print("🗣️  Speaking response...")
        
        with wave.open("response.wav", "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        
        # Play the synthesized audio
        audio_ctrl.play_audio_thread("response.wav")
        
        # Wait a bit longer to ensure audio finishes
        # Estimate speaking time (rough calculation: ~150 words per minute)
        estimated_duration = len(text.split()) / 2.5  # ~150 words/min = 2.5 words/sec
        wait_time = max(2.0, estimated_duration + 1.0)  # At least 2 seconds, plus buffer
        
        print(f"⏳ Waiting {wait_time:.1f}s for speech to complete...")
        time.sleep(wait_time)
        
    finally:
        wake_detector.is_speaking = False
        print("✅ Speech complete, resuming wake word detection")
        
def detect_wake_word(audio_data, wake_words=None):
    """Enhanced wake word detection with multiple wake words"""
    if wake_words is None:
        wake_words = WAKE_WORDS
    
    if isinstance(audio_data, str):
        text = audio_data.lower().strip()
    else:
        # Assume it's audio data that needs transcription
        return False
    
    # Check for exact matches and partial matches
    for wake_word in wake_words:
        wake_word = wake_word.lower()
        if wake_word in text:
            print(f"   Wake word detected: '{wake_word}' in '{text}'")
            return True
    
    # Check for phonetically similar words if enabled
    if ENABLE_SIMILAR_WORDS:
        for wake_word in wake_words:
            wake_word_lower = wake_word.lower()
            if wake_word_lower in SIMILAR_WORDS:
                for similar in SIMILAR_WORDS[wake_word_lower]:
                    if similar in text:
                        print(f"   Similar wake word detected: '{similar}' for '{wake_word}' in '{text}'")
                        return True
    
    return False

def record_command_audio(max_duration=None):
    """Record audio after wake word detection with dynamic stopping"""
    if max_duration is None:
        max_duration = MAX_RECORDING_DURATION
        
    print("🎤 Recording command... (speak now)")
    
    audio_buffer = []
    silence_start = None
    recording = True
    has_detected_voice = False  # Track if we've heard any voice yet
    
    # Use the calibrated threshold from wake word detection for consistency
    if wake_detector.calibrated_threshold is not None:
        # voice_threshold = wake_detector.calibrated_threshold * 0.7  # Slightly lower for command detection
        voice_threshold = wake_detector.calibrated_threshold     # fix later 
    else:
        voice_threshold = 0.03  # Fallback threshold
    
    def audio_callback(indata, frames, time, status):
        nonlocal audio_buffer, silence_start, recording, has_detected_voice
        if recording:
            audio_buffer.append(indata.copy())
            
            # Check for silence using consistent energy calculation
            energy = np.sqrt(np.mean(indata**2))
            
            if energy > voice_threshold:
                has_detected_voice = True
                silence_start = None  # Reset silence timer
                # Debug: Show voice detection during command recording (less frequent)
                if len(audio_buffer) % 50 == 0:  # Much less frequent feedback
                    print(f"🎙️  Voice: {energy:.3f} > {voice_threshold:.3f}")
            else:
                # Only start silence timer if we've detected voice before
                if has_detected_voice and silence_start is None:
                    silence_start = time.inputBufferAdcTime
    
    try:
        with sd.InputStream(callback=audio_callback,
                          channels=1,
                          samplerate=AUDIO_SAMPLE_RATE,
                          dtype=np.float32):
            
            start_time = time.time()
            while recording and (time.time() - start_time) < max_duration:
                time.sleep(0.1)
                
                # Only stop on silence if we've detected voice AND had enough silence
                # Use the updated SILENCE_THRESHOLD from config
                if (has_detected_voice and silence_start and 
                    (time.time() - silence_start) > SILENCE_THRESHOLD):
                    recording = False
                    print("  S ilence detected, stopping recording")
                    break
        
        if audio_buffer:
            # Combine all audio chunks
            audio_data = np.concatenate(audio_buffer, axis=0)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
                audio_int16 = (audio_data * 32767).astype(np.int16)
                
                with wave.open(tmpfile.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(AUDIO_SAMPLE_RATE)
                    wf.writeframes(audio_int16.tobytes())
                
                duration = len(audio_data) / AUDIO_SAMPLE_RATE
                print(f"✅ Recording complete ({duration:.1f}s)")
                return tmpfile.name
    
    except Exception as e:
        print(f"Error recording command: {e}")
        return None
    
    return None

def start_assistant():
    """Main assistant loop with wake word detection"""
    print("🤖 Jetson Assistant Starting...")
    print(f"  Wake words: {WAKE_WORDS}")
    print(f"  Acknowledgment sound: {'Enabled' if ENABLE_ACKNOWLEDGMENT_SOUND else 'Disabled'}")
    
    # Test Ollama connection first
    try:
        test_response = ollama.chat(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': 'Hello'}],
            options={'timeout': 10}
        )
        print("  Ollama connection verified")
    except Exception as e:
        print(f"  ⚠️Ollama connection issue: {e}")
        print("   Make sure Ollama is running and model is available")
    
    # Calibrate environment
    try:
        wake_detector.calibrate_environment()
        
        # Ask user if they want to manually adjust threshold
        print("\n  If the threshold seems wrong, you can manually override it.")
        print("  Press Enter to continue, or type a threshold value (e.g., 0.003):")
        user_input = input().strip()
        
        if user_input:
            try:
                manual_threshold = float(user_input)
                if 0.001 <= manual_threshold <= 0.5:
                    wake_detector.calibrated_threshold = manual_threshold
                    print(f"  Manual threshold set to {manual_threshold:.3f}")
                else:
                    print("  Invalid threshold range. Using auto-calibrated value.")
            except ValueError:
                print("  ⚠️ Invalid threshold format. Using auto-calibrated value.")

    except Exception as e:
        print(f"  ⚠️ Calibration failed: {e}")
        print("   Continuing with default threshold...")
        wake_detector.calibrated_threshold = 0.03  # Conservative fallback
    
    print("🚀 Assistant ready! Say one of the wake words to activate.")
    print(f"🎯 Current threshold: {wake_detector.calibrated_threshold:.3f}")
    
    consecutive_failures = 0
    max_failures = 3
    
    while True:
        try:
            # Reset stop event
            wake_detector.stop_listening.clear()
            
            # Listen for wake word with timeout
            print("👂 Listening for wake word...")
            wake_word_detected = wake_detector.listen_for_wake_word()
            
            if wake_word_detected:
                consecutive_failures = 0  # Reset failure counter
                
                # Record command after wake word
                command_file = record_command_audio()
                
                if command_file:
                    try:
                        # Transcribe the command
                        transcribed_text = transcribe_audio(command_file)
                        print(f"🗣️  Command: {transcribed_text}")
                        
                        if transcribed_text.strip():
                            # Process with LLM
                            response = ask_llm(transcribed_text)
                            print(f"🤖 Response: {response}")
                            
                            # Speak response
                            if response:
                                text_to_speech(response)
                        else:
                            print("⚠️  No command detected")
                        
                        # Clean up temp file
                        os.unlink(command_file)
                        
                    except Exception as e:
                        print(f"Error processing command: {e}")
                        text_to_speech(ERROR_RESPONSE)
                        try:
                            os.unlink(command_file)
                        except:
                            pass
                
                # Brief pause before listening again
                time.sleep(1)
            
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print(f"⚠️  {max_failures} consecutive listening failures, resetting...")
                    consecutive_failures = 0
                    time.sleep(2)
            
        except KeyboardInterrupt:
            print("\n🛑 Assistant stopping...")
            wake_detector.stop_listening.set()
            break
        except Exception as e:
            print(f"Error in assistant loop: {e}")
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                print("⚠️  Too many errors, restarting assistant loop...")
                consecutive_failures = 0
                time.sleep(3)
            else:
                time.sleep(1)
        
        
from numpy import int16

if __name__ == "__main__":
    # Start the assistant
    start_assistant()

# Example usage functions for testing individual components
def test_wake_word_detection():
    """Test wake word detection with sample audio"""
    test_phrases = [
        "Hey Jetson, what time is it?",
        "Jetson turn on the lights",
        "Hello Jetson",
        "This is not a wake word",
        "Hey there friend"
    ]
    
    for phrase in test_phrases:
        result = detect_wake_word(phrase)
        print(f"'{phrase}' -> Wake word detected: {result}")

def test_single_recording():
    """Test recording and transcription without wake word detection"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
        record_audio(tmpfile.name, duration=5)
        transcribed_text = transcribe_audio(tmpfile.name)
        print(f"Transcribed: {transcribed_text}")
        
        response = ask_llm(transcribed_text)
        print(f"Response: {response}")
        
        if response:
            text_to_speech(response)
        
        os.unlink(tmpfile.name)
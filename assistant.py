import os
import tempfile
import wave
import threading
import time
import queue
import yaml

import numpy as np
import sounddevice as sd
from scipy.signal import butter, filtfilt
from collections import deque

import ollama
from piper import PiperVoice
import whisper

import audio_ctrl

# Load configuration
def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), 'assistant_config.yaml')
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Warning: Could not load config file: {e}")
        return {}

config = load_config()

# Voice and transcription models
voice = PiperVoice.load("voices/glados.onnx", "voices/glados.json")
whisper_model = whisper.load_model("base")

# Configuration settings
WAKE_WORDS = config.get('wake_words', ["rover", "hey rover", "hello rover"])
AUDIO_SAMPLE_RATE = config.get('audio', {}).get('sample_rate', 16000)
CHUNK_SIZE = config.get('audio', {}).get('chunk_size', 1024)
ENERGY_THRESHOLD = config.get('audio', {}).get('energy_threshold', 500)
SILENCE_THRESHOLD = config.get('audio', {}).get('silence_threshold', 2.0)
MAX_RECORDING_DURATION = config.get('audio', {}).get('max_recording_duration', 15)
CALIBRATION_DURATION = config.get('audio', {}).get('calibration_duration', 3)
ENABLE_ACKNOWLEDGMENT_SOUND = config.get('assistant', {}).get('enable_acknowledgment_sound', True)
ACKNOWLEDGMENT_SOUND = config.get('assistant', {}).get('acknowledgment_sound', "connected/connected.mp3")
ERROR_RESPONSE = config.get('assistant', {}).get('error_response', "Sorry, I didn't understand that command.")

ENABLE_SIMILAR_WORDS = config.get('detection', {}).get('enable_similar_words', True)
SIMILAR_WORDS = config.get('detection', {}).get('similar_words', {
    "rover": ["rower", "rofer", "rovor", "rover's", "robes"],
    "hey rover": ["a rover", "hay rover", "hey rower", "hey rofer", "hey rovor", "hey rovers", "hey robert"],
    "hello rover": ["hello rower", "hello rofer", "halo rover", "hello rovor", "hello rovers", "hello robert"]
})
LLM_MODEL = config.get('llm', {}).get('model', 'gemma2:2b')
LLM_BASE_URL = config.get('llm', {}).get('base_url', 'http://localhost:11434')
LLM_TIMEOUT = config.get('llm', {}).get('timeout', 30)


class WakeWordDetector:
    def __init__(self):
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.stop_listening = threading.Event()
        self.energy_buffer = deque(maxlen=50)
        self.calibrated_threshold = None
        self.is_speaking = False

    def calculate_audio_energy(self, audio_data):
        return np.sqrt(np.mean(audio_data**2))

    def preprocess_audio(self, audio_data):
        nyquist = AUDIO_SAMPLE_RATE * 0.5
        low = 300 / nyquist
        high = 3400 / nyquist
        b, a = butter(1, [low, high], btype='band')
        return filtfilt(b, a, audio_data)

    def calibrate_environment(self, duration=None):
        if duration is None:
            duration = CALIBRATION_DURATION

        print("Calibrating background noise... Please stay quiet.")
        audio = sd.rec(int(duration * AUDIO_SAMPLE_RATE),
                       samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype=np.float32)
        sd.wait()

        background_energy = self.calculate_audio_energy(audio.flatten())

        # TODO: Implement better calibration logic
        # if background_energy > 0.1:
        #     self.calibrated_threshold = 0.05
        #     print(f"High background noise ({background_energy:.3f}), using low threshold")
        # elif background_energy > 0.05:
        #     self.calibrated_threshold = 0.03
        #     print(f"Medium background noise ({background_energy:.3f}), using moderate threshold")
        # else:
        #     self.calibrated_threshold = 0.007
        #     print(f"Low background noise ({background_energy:.3f}), using sensitive threshold")
        
        # TODO: Hardcoded threshold for now until I can fix the calibration logic
        self.calibrated_threshold = 0.007 # Default threshold for quiet environments 
        print(f"Background energy: {background_energy:.3f}, Threshold: {self.calibrated_threshold:.3f}")

    def detect_voice_activity(self, audio_chunk):
        energy = self.calculate_audio_energy(audio_chunk)
        self.energy_buffer.append(energy)

        if self.calibrated_threshold is not None:
            threshold = self.calibrated_threshold
        else:
            # 32767.0 is max value for int16 audio and is used to convert between
            # integer audio samples and floating-point representations
            threshold = ENERGY_THRESHOLD / 32767.0 

        voice_detected = energy > threshold

        if len(self.energy_buffer) > 3:
            recent_avg = np.mean(list(self.energy_buffer)[-3:])
            if recent_avg > threshold * 0.8:
                voice_detected = True

        return voice_detected

    # TODO: Implement a more robust wake word detection algorithm
    # Currently, this method is a placeholder and should be replaced with a proper wake word detection
    # algorithm or library that can handle wake word detection more effectively.
    def listen_for_wake_word(self):
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}")
            try:
                self.audio_queue.put(indata.copy(), timeout=1.0)
            except queue.Full:
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
                    if self.is_speaking:
                        time.sleep(0.1)
                        continue

                    if time.time() - start_time > 10:
                        print("Listening timeout, resetting...")
                        break

                    try:
                        chunk = self.audio_queue.get(timeout=0.5).flatten()
                        processed = self.preprocess_audio(chunk)
                        has_voice = self.detect_voice_activity(processed)

                        if has_voice:
                            last_activity_time = time.time()
                            recording_triggered = True
                            audio_buffer.extend(processed)
                            max_samples = AUDIO_SAMPLE_RATE * 5
                            if len(audio_buffer) > max_samples:
                                audio_buffer = audio_buffer[-max_samples:]

                        silence_duration = time.time() - last_activity_time

                        # There is definitely a better way to handle this
                        if recording_triggered and silence_duration > SILENCE_THRESHOLD:
                            if len(audio_buffer) > AUDIO_SAMPLE_RATE * 0.5:
                                print(f"Processing {len(audio_buffer)/AUDIO_SAMPLE_RATE:.1f}s of audio...")
                                if self.process_audio_for_wake_word(np.array(audio_buffer)):
                                    if ENABLE_ACKNOWLEDGMENT_SOUND:
                                        print("Playing acknowledgment sound...")
                                        try:
                                            audio_ctrl.play_file(ACKNOWLEDGMENT_SOUND)
                                            time.sleep(0.5)
                                        except:
                                            print("*acknowledgment sound*")
                                    return True

                            audio_buffer = []
                            recording_triggered = False

                    except queue.Empty:
                        continue
                    except Exception as e:
                        print(f"Error in wake word detection: {e}")
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
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
                audio_int16 = (audio_data * 32767).astype(np.int16)
                with wave.open(tmpfile.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(AUDIO_SAMPLE_RATE)
                    wf.writeframes(audio_int16.tobytes())

                text = transcribe_audio(tmpfile.name)
                print(f"Heard: '{text}'")
                os.unlink(tmpfile.name)
                return detect_wake_word(text)

        except Exception as e:
            print(f"Error processing audio for wake word: {e}")
            return False

wake_detector = WakeWordDetector() # Initialize the wake word detector


def record_audio(filename, duration=5, fs=16000):
    print("Recording audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())
    print("Recording complete.")


def transcribe_audio(filename):
    result = whisper_model.transcribe(filename, language='en')
    return result.get('text', '').strip()


def ask_llm(query):
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {'role': 'system', 'content': "Your name is Jetson, an autonomous rover that can interact with the world by accepting commands from the user. Only output pure raw text."},
            {'role': 'user', 'content': query}
        ],
    )
    return response['message']['content']


def text_to_speech(text):
    try:
        wake_detector.is_speaking = True
        print("Speaking response...")
        with wave.open("response.wav", "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        audio_ctrl.play_audio_thread("response.wav")
        # Estimate duration: ~150 words/min = 2.5 words/sec
        estimated_duration = len(text.split()) / 2.5
        time.sleep(max(2.0, estimated_duration + 1.0))
    finally:
        wake_detector.is_speaking = False
        print("Speech complete.")


def detect_wake_word(text, wake_words=None):
    if wake_words is None:
        wake_words = WAKE_WORDS

    text_lower = text.lower().strip()
    for wake_word in wake_words:
        if wake_word.lower() in text_lower:
            print(f"Wake word detected: '{wake_word}'")
            return True

    if ENABLE_SIMILAR_WORDS:
        for wake_word, similars in SIMILAR_WORDS.items():
            for similar in similars:
                if similar in text_lower:
                    print(f"Similar wake word detected: '{similar}' for '{wake_word}'")
                    return True

    return False


def record_command_audio(max_duration=None):
    if max_duration is None:
        max_duration = MAX_RECORDING_DURATION

    print("Recording command...")
    audio_buffer = []
    silence_start = None
    recording = True
    has_detected_voice = False

    if wake_detector.calibrated_threshold is not None:
        voice_threshold = wake_detector.calibrated_threshold
    else:
        voice_threshold = 0.03

    def audio_callback(indata, frames, time_info, status):
        nonlocal audio_buffer, silence_start, recording, has_detected_voice
        if recording:
            audio_buffer.append(indata.copy())
            energy = np.sqrt(np.mean(indata**2))
            if energy > voice_threshold:
                has_detected_voice = True
                silence_start = None
            else:
                if has_detected_voice and silence_start is None:
                    silence_start = time.time()

    try:
        with sd.InputStream(callback=audio_callback,
                            channels=1,
                            samplerate=AUDIO_SAMPLE_RATE,
                            dtype=np.float32):

            start_time = time.time()
            # TODO: Implement a more robust way to handle recording duration and silence detection
            while recording and (time.time() - start_time) < max_duration:
                time.sleep(0.1)
                if has_detected_voice and silence_start is not None:
                    if (time.time() - silence_start) > SILENCE_THRESHOLD:
                        recording = False
                        print("Silence detected, stopping recording")
                        break

        if audio_buffer:
            audio_data = np.concatenate(audio_buffer, axis=0)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
                audio_int16 = (audio_data * 32767).astype(np.int16)
                with wave.open(tmpfile.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(AUDIO_SAMPLE_RATE)
                    wf.writeframes(audio_int16.tobytes())

                duration = len(audio_data) / AUDIO_SAMPLE_RATE
                print(f"Recording complete ({duration:.1f}s)")
                return tmpfile.name

    except Exception as e:
        print(f"Error recording command: {e}")
        return None

    return None


def start_assistant():
    print("Jetson Assistant Starting...")
    print(f"Wake words: {WAKE_WORDS}")
    print(f"Acknowledgment sound: {'Enabled' if ENABLE_ACKNOWLEDGMENT_SOUND else 'Disabled'}")

    # Test LLM connection
    try:
        _ = ollama.chat(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': 'Hello'}],
            options={'timeout': 10}
        )
        print("LLM connection verified")
    except Exception as e:
        print(f"LLM connection issue: {e}")
        print("Make sure the LLM service is running and the model is available")

    # Calibrate environment
    try:
        wake_detector.calibrate_environment()
        print("If the threshold seems wrong, you can manually override it.")
        manual = input("Press Enter to continue or type a threshold value (e.g., 0.003): ").strip()
        if manual:
            try:
                value = float(manual)
                if 0.001 <= value <= 0.5:
                    wake_detector.calibrated_threshold = value
                    print(f"Manual threshold set to {value:.3f}")
                else:
                    print("Invalid threshold range. Using auto-calibrated value.")
            except ValueError:
                print("Invalid format. Using auto-calibrated value.")
    except Exception as e:
        print(f"Calibration failed: {e}")
        wake_detector.calibrated_threshold = 0.03

    print("Assistant ready!")
    print(f"Current threshold: {wake_detector.calibrated_threshold:.3f}")

    consecutive_failures = 0
    max_failures = 3

    while True:
        try:
            wake_detector.stop_listening.clear()
            print("Listening for wake word...")
            if wake_detector.listen_for_wake_word():
                consecutive_failures = 0
                command_file = record_command_audio()
                if command_file:
                    try:
                        command_text = transcribe_audio(command_file)
                        print(f"Command: {command_text}")
                        if command_text:
                            response = ask_llm(command_text)
                            print(f"Response: {response}")
                            if response:
                                text_to_speech(response)
                        else:
                            print("No command detected")
                    except Exception as e:
                        print(f"Error processing command: {e}")
                        text_to_speech(ERROR_RESPONSE)
                    finally:
                        try:
                            os.unlink(command_file)
                        except:
                            pass
                time.sleep(1)
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print("Multiple listening failures. Threshold might be too high. Resetting...")
                    consecutive_failures = 0
                    time.sleep(2)
        except KeyboardInterrupt:
            # Graceful shutdown on keyboard interrupt (^C)
            print("Assistant stopping...")
            wake_detector.stop_listening.set()
            break
        except Exception as e:
            print(f"Error in assistant loop: {e}")
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                print("Too many errors, restarting loop...")
                consecutive_failures = 0
                time.sleep(3)
            else:
                time.sleep(1)

from numpy import int16

if __name__ == "__main__":
    start_assistant()
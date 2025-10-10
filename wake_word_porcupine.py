import numpy as np
import pvporcupine
import speech_recognition as sr
import time

# --- CONFIG ---
DEFAULT_SAMPLE_RATE = 16000
WAKEWORD = "porcupine"
ACCESS_KEY = 'DqiDIJH2kU5J1HIu0+TbZKaghKCU3ahwBUpWOzVXqyurokHNTgQiQQ==' # MUST be a real key for Porcupine

class WakeWordDetector:
    def __init__(self, use_porcupine=True):
        
        # Initialize Recognizer and Mic Source
        self.r = sr.Recognizer()
        
        # Use sr.Microphone to get device information
        mic_list = sr.Microphone.list_microphone_names()
        # You'd need custom logic here to pick the best mic index from mic_list 
        # For simplicity, we'll try to use the system default (index=None)
        
        try:
             # Initialize Microphone object
            self.mic_source = sr.Microphone(device_index=None, sample_rate=DEFAULT_SAMPLE_RATE)
            if self.mic_source.device_index is not None:
                mic_name = mic_list[self.mic_source.device_index]
                print(f"Using microphone: {self.mic_source.device_index} ({mic_name})")
            else:
                # Fallback if device_index is None even after successful init
                print("Using default microphone, but index name could not be retrieved.")
                
        except Exception as e:
            print(f"Failed to initialize PyAudio Microphone: {e}")
            raise

        # Calibration Step
        print("Starting ambient noise calibration (speak after 1s)...")
        with self.mic_source as source:
            # This method calculates the appropriate energy_threshold automatically
            self.r.adjust_for_ambient_noise(source, duration=3) 
            self.r.pause_threshold = 0.5 
            self.r.energy_threshold *= 1.5 # Optional: Set a safety margin above ambient
            
        self.calibrated_threshold = self.r.energy_threshold
        print(f"Calibrated energy threshold: {self.calibrated_threshold:.2f}")

        # Initialize Porcupine
        self.use_porcupine = use_porcupine and (ACCESS_KEY is not None)
        if self.use_porcupine:
            self.porcupine = pvporcupine.create(
                access_key=ACCESS_KEY,
                keywords=[WAKEWORD]
            )
            self.frame_length = self.porcupine.frame_length
            self.frame_duration_ms = self.frame_length / DEFAULT_SAMPLE_RATE * 1000
            print(f"Porcupine initialized. Frame duration: {self.frame_duration_ms:.1f}ms")
        
    def listen(self):
        print("\nListening for wake word...")
        
        # Start background listening with VAD
        # We use listen_in_background to process the stream continuously
        
        def callback(recognizer, audio):
            # The audio object contains the buffered stream audio ready for processing
            if self.use_porcupine:
                self.process_porcupine(audio)
            else:
                # VAD-only: If we get here, speech was detected (VAD triggered)
                print("Voice Activity Detected! (VAD-only mode)")
                self.capture_command(audio)
        
        # This starts a background thread and calls 'callback' when speech is detected (VAD)
        self.stop_listening_sr = self.r.listen_in_background(self.mic_source, callback)
        
        # Wait until the user presses Ctrl+C
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_listening_sr(wait_for_stop=False)
            if self.use_porcupine:
                self.porcupine.delete()
            print("\nExiting program.")

    def process_porcupine(self, audio):
        """Processes audio chunks specifically for Porcupine detection."""
        
        # NOTE: sr.listen_in_background buffers audio when VAD detects speech.
        # This is a key difference. We are only checking Porcupine *after* VAD.
        
        # Convert audio object to raw int16 data
        raw_data = audio.get_raw_data(convert_rate=DEFAULT_SAMPLE_RATE, convert_width=2)
        audio_frame_size = self.frame_length * 2  # 2 bytes per int16 sample
        
        # Check frames against Porcupine's frame length
        for i in range(0, len(raw_data) - audio_frame_size, audio_frame_size):
            frame = raw_data[i:i + audio_frame_size]
            
            # Porcupine processes raw bytes, must be converted to numpy int16 array
            pcm = np.frombuffer(frame, dtype=np.int16) 
            result = self.porcupine.process(pcm)
            
            if result >= 0:
                print("✅ Wake word detected!")
                self.capture_command(audio)
                return

    def capture_command(self, audio):
        """Transcribes the command from the final AudioData object."""
        print("🗣️ Processing command...")
        
        try:
            # Transcribe the buffered AudioData
            command = self.r.recognize_google(audio)
            
            if command:
                print(f"👂 Transcribed command: '{command}'")
                # TODO: Add command execution logic
            else:
                print("No clear command detected.")
                
        # ... (Exception handling remains the same) ...
        except sr.UnknownValueError:
            print("Could not understand audio.")
        except sr.RequestError as e:
            print(f"Service error: {e}")
            
        print("Returning to wake word listening.")


if __name__ == "__main__":
    detector = WakeWordDetector(use_porcupine=True)
    detector.listen()

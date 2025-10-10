import vosk
import json
import time
import sounddevice as sd
import sys
import queue
import numpy as np
from ollama import chat
from collections import deque
from audio_config import parse_audio_args, init_tts_engine # The wake word to listen for 

class WakeWordDetector:
    def __init__(self):
        
        # Parse audio device and sample rate from command line
        try:
            self.device_index, self.samplerate, self.wakeword, self.model_path = parse_audio_args()
            self.model = vosk.Model(self.model_path)
        except Exception as e:
            print(f"Error parsing audio args: {e}")
            sys.exit(1)
        self.running = True
        self.audio_queue = queue.Queue()

        
        print("Vosk model loaded successfully.")

        if not self.samplerate:
            self.samplerate = 16000
        
        self.blocksize = self.samplerate // 4

        print(f"Using audio device index: {self.device_index}, Sample rate: {self.samplerate}")
    
    def callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        # Put the raw audio data into the queue
        self.audio_queue.put(bytes(indata))

    def listen_for_wake_word(self):
        """Listens continuously using sounddevice stream and processes with Vosk."""
        
        print(f"\nListening for wake word (sounddevice stream mode)... Say '{self.wakeword}' to activate.")
        
        # Open the continuous, non-blocking audio stream
        try:
            with sd.RawInputStream(samplerate=self.samplerate, blocksize=self.blocksize, 
            device=self.device_index, dtype="int16", channels=1, callback=self.callback):

                # Inform user
                print("#" * 80)
                print("Press Ctrl+C to stop the recording")
                print("#" * 80)

                # Initialize Vosk recognizer
                rec = vosk.KaldiRecognizer(self.model, self.samplerate)
                while self.running:

                    # Get audio data from the queue
                    try:
                        data = self.audio_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        text = result.get("text", "")
                        if self.wakeword in text.lower():
                            print(f"Wake word '{self.wakeword}' detected!")
                            command = self.capture_command(rec)
                            if command:
                                self.respond_to_command(command)
                                print(f"\nListening for wake word (sounddevice stream mode)... Say '{self.wakeword}' to activate.")
                            else:
                                print("No command captured after wake word.")

        except KeyboardInterrupt:
            print("\nStopping wake word detection.")
            self.running = False

    def capture_command(self, rec):
        """
        Captures the command immediately after wake word using:
        - Pre-roll buffer to avoid missing the start
        - Single audio queue + flag to minimize complexity
        """
        print("🗣️ Listening for command...")

        PRE_ROLL_SECONDS = 0.5     # keep last 0.5s of audio
        COMMAND_TIMEOUT = 8        # max command length in seconds       
        SILENCE_TIMEOUT = 1        # stop early if no audio for this many seconds             # must match stream blocksize

        # Pre-roll buffer
        pre_roll_chunks = deque(maxlen=int(self.samplerate * PRE_ROLL_SECONDS / self.blocksize))

        # Fill pre-roll with existing chunks in the queue
        while not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get_nowait()
                pre_roll_chunks.append(chunk)
            except queue.Empty:
                break

        # Initialize capture
        command_start_time = time.time()
        last_audio_time = command_start_time
        all_chunks = list(pre_roll_chunks)

        # Capture loop
        while self.running:
            # Check for timeout
            elapsed = time.time() - command_start_time
            if elapsed > COMMAND_TIMEOUT:
                break
            elif time.time() - last_audio_time > SILENCE_TIMEOUT:
                break

            try:
                chunk = self.audio_queue.get(timeout=0.05)
                rec.AcceptWaveform(chunk)
                last_audio_time = time.time()

            except queue.Empty:
                pass

            # Get final transcription
        result = json.loads(rec.FinalResult())
        command_text = result.get("text", "").strip()
        words = command_text.split()
        if words and words[-1] in ["the", "a", "an"]:
            words = words[:-1]
        command_text = " ".join(words)

        if command_text:
            print(f"👂 Transcribed command: '{command_text}'")
            return command_text
        else:
            print("No clear command detected.")
            command_text = None
            return command_text

        print("Returning to wake word listening.\n" + "="*40)
    
    def respond_to_command(self, command_text):
        """Send recognized command to Ollama and speak the reply."""
        print("🤖 Sending to Ollama...")
        try:
            response = chat(model='llama3.2', messages=[
                {'role': 'user', 'content': command_text}
            ])
            reply = response['message']['content']
            print(f"Ollama reply: {reply}")

            # Stop Microphone to avoid feedback
            sd.stop()

            # Speak it
            engine = init_tts_engine()
            engine.say(reply)
            engine.runAndWait()
            engine.stop()

        except Exception as e:
            print(f"Error communicating with Ollama: {e}")

if __name__ == "__main__":
    detector = WakeWordDetector()
    detector.listen_for_wake_word()
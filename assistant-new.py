import whisper
from piper import PiperVoice
import speech_recognition as sr
from os import getenv

voice = PiperVoice.load("voices/glados.onnx", "voices/glados.json")
whisper_model = whisper.load_model("base")

# define the assistant class
class Assistant:
    def __init__(self):
        self.voice = voice
        self.whisper_model = whisper_model

    def ask_llm(self, question):
        # Implement LLM asking functionality
        pass

    def record_audio(self):
        # Implement audio recording functionality
        pass
    
    def listen(self):
        # Implement listening functionality using Porcupine
        pass

    def respond(self, text):
        # Implement response functionality using PiperVoice
        pass
    
    
    def transcribe_audio(self, audio):
        # Implement audio transcription using Whisper
        pass
    
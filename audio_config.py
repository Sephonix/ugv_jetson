#!/usr/bin/env python3
"""
audio_config.py

This module defines a simple CLI helper for selecting
audio input devices, sample rates, wake words, and Vosk model paths
for microphone-based applications.
"""

import argparse
import sounddevice as sd
import pyttsx3
import numpy as np

def int_or_str(value):
    """Helper for argparse to handle numeric or string device IDs."""
    try:
        return int(value)
    except ValueError:
        return value


def parse_audio_args():
    """
    Parse command line arguments related to audio device configuration.
    Returns:
        (device_index, samplerate, wakeword, model_path)
    """
    parser = argparse.ArgumentParser(
        description="Audio input configuration helper for Vosk applications."
    )

    # Audio device arguments
    parser.add_argument(
        "-d", "--device",
        type=int_or_str,
        help="Input device ID or substring name (use --list-devices to view).",
    )

    parser.add_argument(
        "-r", "--samplerate",
        type=int,
        help="Sampling rate in Hz (default: use device's default rate).",
    )

    parser.add_argument(
        "-l", "--list-devices",
        action="store_true",
        help="Show list of audio devices and exit.",
    )

    # wake word argument
    parser.add_argument(
        "-w", "--wakeword",
        type=str,
        default="banana",
        help="Wake word to listen for (default: 'banana')."
    )

    # Vosk model path argument
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="vosk-model-en-us-0.22-lgraph",
        help="Path to Vosk model directory (default: 'vosk-model-en-us-0.22-lgraph')."
    )

    args = parser.parse_args()

    # Handle --list-devices early
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)

    # Determine samplerate if not specified
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, "input")
        args.samplerate = int(device_info["default_samplerate"])

    return args.device, args.samplerate, args.wakeword, args.model

def init_tts_engine(rate: int = 175, volume: float = 1.0, voice: str = None):
    """
    Initialize and return a pyttsx3 TTS engine.
    You can customize rate, volume, and voice name.
    """
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    engine.setProperty('volume', volume)

    # Optional: set specific voice (male/female)
    if voice:
        voices = engine.getProperty('voices')
        for v in voices:
            if voice.lower() in v.name.lower():
                engine.setProperty('voice', v.id)
                break
    return engine

# Optional: allow running this file directly for quick testing
if __name__ == "__main__":
    device, rate, wakeword, model_path = parse_audio_args()
    print(f"Selected input device: {device}")
    print(f"Sample rate: {rate} Hz")
    print(f"Wake word: '{wakeword}'")
    print(f"Model path: '{model_path}'")
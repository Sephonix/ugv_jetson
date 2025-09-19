# Jetson Assistant - Modular Architecture

This is a refactored version of the Jetson Assistant with a clean, maintainable modular architecture.

## Directory Structure

```
ugv_jetson/
├── assistant_refactored.py     # Main application entry point
├── assistant.py                # Original monolithic version
├── wake_word_config.yaml       # Configuration file
├── assistant_modules/          # Modular components
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── audio/                    # Audio processing and Speech processing
│   │   ├── __init__.py
│   │   ├── processor.py            # Audio utilities and calibration
│   │   └── recorder.py             # Dynamic audio recording
│   │   ├── transcription.py        # Whisper-based speech-to-text
│   │   └── synthesis.py            # Piper-based text-to-speech
│   ├── detection/                # Wake word detection
│   │   ├── __init__.py
│   │   ├── wake_word.py            # Wake word matching logic
│   │   └── detector.py             # Main wake word detector
│   └── llm/                      # Language model integration
│       ├── __init__.py
│       └── client.py               # Ollama LLM client
├── voices/                     # Voice models
├── sounds/                     # Audio files
└── other files...
```

## Module Responsibilities

### `config.py`
- Centralized configuration management
- YAML file loading with fallback defaults
- Type-safe property accessors

### `audio/`
- **`processor.py`**: Audio utilities, preprocessing, calibration, energy calculation
- **`recorder.py`**: Dynamic recording with voice activity detection
- **`transcription.py`**: Whisper-based speech-to-text
- **`synthesis.py`**: Piper-based text-to-speech with audio playback

### `detection/`
- **`wake_word.py`**: Wake word matching logic (exact and similar words)
- **`detector.py`**: Main wake word detection with audio streaming

### `llm/`
- **`client.py`**: Ollama LLM client for generating responses

## Usage

Run the refactored assistant:
```bash
python assistant_refactored.py
```

## Testing

The main file includes test functions:
- `test_components()`: Test wake word detection
- `test_single_recording()`: Test recording/transcription without wake words

## Benefits of This Architecture

1. **Separation of Concerns**: Each module has a single responsibility
2. **Easier Testing**: Components can be tested independently
3. **Maintainability**: Changes to one component don't affect others
4. **Reusability**: Modules can be imported and used separately
5. **Configuration**: Centralized config management
6. **Error Isolation**: Failures in one module don't crash the entire system

## Migration

The original `assistant.py` file is preserved. The new modular version in `assistant_refactored.py` provides the same functionality with better organization.

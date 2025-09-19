import os
import yaml

CONFIG_PATH = os.getenv("CONFIG_PATH", "assistant_config.yaml")
    

class Settings:
    def __init__(self):
        # Load settings from a configuration file
        self.LLM_MODEL = ""
        self.LLM_BASE_URL = ""
        self.LLM_TIMEOUT = 60
        
    def load_config(self):
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as file:
                config = yaml.safe_load(file)
                self.LLM_MODEL = config.get("llm").get("model")
                self.LLM_BASE_URL = config.get("llm").get("base_url")
                self.LLM_TIMEOUT = config.get("llm").get("timeout")
        else:
            raise FileNotFoundError(f"Configuration file {CONFIG_PATH} not found.")


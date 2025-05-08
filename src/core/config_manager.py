import yaml
import json
import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConfigManager:
    def __init__(self, config_dict=None):
        self.config_dict = config_dict or {}

    def get(self, key, default=None):
        return self.config_dict.get(key, default)

    @staticmethod
    def load_file(config_file: str):
        """Loads configuration from a YAML or JSON file."""
        try:
            with open(config_file, "r") as file:
                if config_file.endswith((".yml", ".yaml")):
                    config_data = yaml.safe_load(file)
                else:
                    config_data = json.load(file)
            return ConfigManager(config_data)
        except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Error loading config file {config_file}: {e}")


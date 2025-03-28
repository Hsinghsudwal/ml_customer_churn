import yaml
import json
import pandas as pd
import pickle
import os
from typing import Dict, Any, Callable, List
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Config:
    def __init__(self, config_dict=None):
        self.config_dict = config_dict or {}

    def get(self, key, default=None):
        return self.config_dict.get(key, default)

    @staticmethod
    def load_file(config_path: str):
        """Loads configuration from a YAML or JSON file."""
        try:
            with open(config_path, "r") as file:
                if config_path.endswith((".yml", ".yaml")):
                    config_data = yaml.safe_load(file)
                else:
                    config_data = json.load(file)
            return Config(config_data)
        except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Error loading config file {config_path}: {e}")


class ArtifactStore:
    """Stores and retrieves intermediate artifacts for the pipeline."""

    def __init__(self, config):
        self.config = config
        self.base_path = self.config.get("folder_path", {}).get(
            "artifacts", {}
        )
        os.makedirs(self.base_path, exist_ok=True)

    def save_artifact(
        self,
        artifact: Any,
        subdir: str,
        name: str,
    ) -> None:
        """Save an artifact in the specified format."""
        artifact_dir = os.path.join(self.base_path, subdir)
        os.makedirs(artifact_dir, exist_ok=True)
        artifact_path = os.path.join(artifact_dir, name)

        try:
            if name.endswith(".pkl"):
                with open(artifact_path, "wb") as f:
                    pickle.dump(artifact, f)
            elif name.endswith(".csv"):
                if isinstance(artifact, pd.DataFrame):
                    artifact.to_csv(artifact_path, index=False)
                else:
                    raise ValueError("CSV format only supports pandas DataFrames.")
            else:
                raise ValueError(f"Unsupported format for {name}")
            logging.info(f"Artifact '{name}' saved to {artifact_path}")
        except Exception as e:
            logging.error(f"Error saving artifact {name}: {e}")

    def load_artifact(
        self,
        subdir: str,
        name: str,
    ):
        """Load an artifact in the specified format."""
        artifact_path = os.path.join(self.base_path, subdir, name)
        try:
            if os.path.exists(artifact_path):
                if name.endswith(".pkl"):
                    with open(artifact_path, "rb") as f:
                        artifact = pickle.load(f)
                elif name.endswith(".csv"):
                    artifact = pd.read_csv(artifact_path)
                else:
                    raise ValueError(f"Unsupported format for {name}")
                logging.info(f"Artifact '{name}' loaded from {artifact_path}")
                return artifact
            else:
                logging.warning(f"Artifact '{name}' not found in {artifact_path}")
                return None
        except Exception as e:
            logging.error(f"Error loading artifact {name}: {e}")
            return None

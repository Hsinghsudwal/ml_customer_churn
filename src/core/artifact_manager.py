import os
import pandas as pd
import pickle
import json
from .config_manager import ConfigManager

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)



class ArtifactManager:
    """Base class for artifact storage"""

    def save(self, artifact, path, name) -> str:
        """Save an artifact to the store"""
        pass

    def load(self, path: str, name):
        """Load an artifact from the store"""
        pass



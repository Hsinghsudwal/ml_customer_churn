# from core.config_artifact_manager import ConfigManager

from src.experiments.model_register_tracking import (
    model_register_tracking,
)
from src.experiments.model_stage import model_staging

from src.experiments.model_production import model_production

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class ExperimentPipeline:
    def __init__(self, config_file: str):
        self.config = config_file
        # self.config = ConfigManager.load_file(config_file)

    def run(self, results, config_file):

        model_register_tracking(results, self.config)
        model_staging(results)
        model_production(results, config_file)

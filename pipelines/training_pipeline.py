from src.core.mlstack import Mlstack
from src.core.config_manager import ConfigManager
from src.core.artifact_manager import ArtifactManager
from integrations.storage import ArtifactStoreFactory

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


import datetime
import uuid
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class TrainingPipeline:
    def __init__(self, data_path: str, config_file: str):
        # Load configuration
        self.config = ConfigManager.load_file(config_file)
        self.artifact_manager = ArtifactStoreFactory.create_store(self.config)
        self.data_path = data_path

    def run(self):
        pipeline_id = str(uuid.uuid4())
        pipeline_name = "Churn Training Pipeline"  # self.config.get("pipeline_name",{})
        start_time = datetime.datetime.now()

        logging.info(
            f"Starting pipeline '{pipeline_name}' with ID: {pipeline_id} for data at {self.data_path}"
        )

        node1 = DataIngestion(self.data_path).data_ingestion
        node2 = DataValidation().data_validation
        node3 = DataTransformation().data_transformation
        node4 = ModelTrainer().model_trainer
        node5 = ModelEvaluation().model_evaluation

        stack = Mlstack(
            pipeline_name,
            self.config.config_dict,
            nodes=[node1, node2, node3, node4, node5],
            artifact_manager=self.artifact_manager,
        )
        results = stack.run(pipeline_id)
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        logging.info(
            f"Pipeline '{pipeline_name}' with ID: {pipeline_id} completed successfully in {duration}."
        )
        print(f"results keys: {results.keys()}")

        return results



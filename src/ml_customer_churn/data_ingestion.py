import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from core.config_artifact_store import ConfigManager, ArtifactManager

# from

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# import prefect
# from prefect import task, Flow


class DataIngestion:

    def __init__(self, data_path: str):
        self.data_path = data_path

    def data_ingestion(
        self, results, config: ConfigManager, artifact_store: ArtifactManager
    ):
        logging.info("Node 1: data ingestion")
        # Define artifact paths and if artifacts exist, return them
        raw_path_dir = config.get("artifact_path", {}).get("raw_path", {})
        raw_train_artifact_name = config.get("artifact_path", {}).get("train", {})
        raw_test_artifact_name = config.get("artifact_path", {}).get("test", {})

        train_data = artifact_store.load(raw_path_dir, raw_train_artifact_name)
        test_data = artifact_store.load(raw_path_dir, raw_test_artifact_name)

        if train_data is not None and test_data is not None:
            logging.info("Existing artifacts found. Skipping data ingestion.")
            return {"train_data": train_data, "test_data": test_data}

        try:
            # Load raw data
            df = pd.read_csv(self.data_path)

            # Split data
            test_size = config.get("base", {}).get("test_size", {})
            train_data, test_data = train_test_split(
                df, test_size=test_size, random_state=42
            )
            # logging.info(
            #     f"Data split complete. Train shape: {train_data.shape}, Test shape: {test_data.shape}"
            # )

            # Save raw artifacts
            artifact_store.save(
                train_data, subdir=raw_path_dir, name=raw_train_artifact_name
            )

            artifact_store.save(
                test_data, subdir=raw_path_dir, name=raw_test_artifact_name
            )

            logging.info("Data ingestion completed")
            return {"train_data": train_data, "test_data": test_data}

        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
from src.core.oi import ArtifactStore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# import prefect
# from prefect import task, Flow


class DataIngestion:
    """Handles data ingestion process"""

    def __init__(self, config):
        self.config = config
        self.artifact_store = ArtifactStore(config)

    def data_ingestion(self, path):
        """split the input data"""
        # Check if artifacts

        raw_path = self.config.get("folder_path", {}).get("raw_path", {})
        raw_train_filename = self.config.get("folder_path", {}).get("train", {})
        raw_test_filename = self.config.get("folder_path", {}).get("test", {})
        test_size = self.config.get("base", {}).get("test_size", 0.2)


        train_data = self.artifact_store.load_artifact(raw_path, raw_train_filename)
        test_data = self.artifact_store.load_artifact(raw_path, raw_test_filename)

        if train_data is not None and test_data is not None:
            logging.info("Loaded raw artifacts from store. Skipping data ingestion.")
            return train_data, test_data
        
        try:
            # Load raw data
            df = pd.read_csv(path)

            # Split data
            test_size = self.config.get("base",{}).get("test_size",{})
            train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
            logging.info(
                f"Data split complete. Train shape: {train_data.shape}, Test shape: {test_data.shape}"
            )

            # raw_path = self.config.get("folder_path", {}).get("raw_path", {})
            # train_filename = self.config.get("folder_path", {}).get("train", {})
            # test_filename = self.config.get("folder_path", {}).get("test", {})

            # Save raw artifacts
            self.artifact_store.save_artifact(
                train_data, subdir=raw_path, name=raw_train_filename 
            )

            self.artifact_store.save_artifact(
                test_data, subdir=raw_path, name=raw_test_filename
            )

            logging.info("Data ingestion completed")
            return train_data, test_data

        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise
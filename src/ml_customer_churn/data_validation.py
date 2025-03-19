import pandas as pd
import os
from scipy.stats import ks_2samp
import logging
from src.core.oi import ArtifactStore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# import prefect
# from prefect import task


class DataValidation:

    def __init__(self, config):
        self.config = config
        self.artifact_store = ArtifactStore(self.config)

    def data_validation(self, train_data, test_data):
 
        val_path = self.config.get("folder_path", {}).get("validate_path", {})
        val_train_filename = self.config.get("folder_path", {}).get("train_val", {})
        val_test_filename = self.config.get("folder_path", {}).get("test_val", {})

        val_train_data = self.artifact_store.load_artifact(val_path,val_train_filename)
        val_test_data = self.artifact_store.load_artifact(val_path, val_test_filename)

        if val_train_data is not None and val_test_data is not None:
            logging.info(
                "Loaded artifacts. Skipping data validation."
            )
            return val_train_data, val_test_data

        try:
            # Perform data validation (e.g., check for data drift)
            for feature in train_data.columns:
                ks_stat, p_value = ks_2samp(train_data[feature], test_data[feature])

                if p_value < 0.05:
                    logger.warning(
                        f"Data drift detected for feature {feature} with p-value: {p_value}"
                    )
                    raise ValueError("Data drift detected. Validation failed.")

            val_train_data = train_data
            val_test_data = test_data

            # Save validation artifacts
            self.artifact_store.save_artifact(
                val_train_data, subdir=val_path, name=val_train_filename 
            )

            self.artifact_store.save_artifact(
                test_data, subdir=val_path, name=val_test_filename
            )

            logging.info("Data validation completed")
            return val_train_data, val_test_data

        except Exception as e:
            logging.error(f"Error during data validation: {e}")
            raise


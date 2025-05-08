import pandas as pd
from scipy.stats import ks_2samp
import logging
from src.core.config_manager import ConfigManager
from src.core.artifact_manager import ArtifactManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class DataValidation:
    def __init__(self):
        pass

    def data_validation(
        self,
        results,
        config: ConfigManager,
        artifact_store: ArtifactManager,
        pipeline_id,
    ):
        logging.info("Node 2: data validate")

        # Define artifact paths
        val_path_dir = config.get("artifact_path", {}).get("validate_path", {})
        val_train_artifact_name = config.get("artifact_path", {}).get("train_val", {})
        val_test_artifact_name = config.get("artifact_path", {}).get("test_val", {})

        # Load existing validation artifacts
        val_train_data = artifact_store.load(val_path_dir, val_train_artifact_name)
        val_test_data = artifact_store.load(val_path_dir, val_test_artifact_name)

        if val_train_data is not None and val_test_data is not None:
            logging.info(
                "Loaded existing validation artifacts. Skipping data validation."
            )
            return {"val_train_data": val_train_data, "val_test_data": val_test_data}

        # Load train and test data from previous results
        train_data = results.get("train_data")
        test_data = results.get("test_data")

        if train_data is None or test_data is None:
            logging.error(
                "Train or test data not found in results. Validation aborted."
            )
            raise ValueError("Missing train or test data.")

        try:
            # Perform data validation (e.g., Kolmogorov-Smirnov test for data drift)
            # logging.info("Starting data validation...")
            for feature in train_data.columns:
                # Ensure the feature exists in both datasets
                if train_data[feature].dtype not in ["int64", "float64"]:
                    continue

                # Ensure the feature exists in both datasets
                # if feature not in test_data.columns:
                #     logging.warning(
                #         f"Feature '{feature}' missing in test data. Skipping."
                #     )
                #     continue

                ks_stat, p_value = ks_2samp(train_data[feature], test_data[feature])

                ks_stat, p_value = ks_2samp(train_data[feature], test_data[feature])

                if p_value < 0.05:
                    logging.warning(
                        f"Data drift detected for feature '{feature}' with p-value: {p_value}"
                    )
                    raise ValueError(f"Data drift detected for feature '{feature}'")

            # Save validation artifacts
            artifact_store.save(
                train_data,
                subdir=val_path_dir,
                name=val_train_artifact_name,
                pipeline_id=pipeline_id,
            )
            artifact_store.save(
                test_data,
                subdir=val_path_dir,
                name=val_test_artifact_name,
                pipeline_id=pipeline_id,
            )

            logging.info("Data validation completed.")
            return {"val_train_data": train_data, "val_test_data": test_data}

        except Exception as e:
            logging.error(f"Error during data validation: {e}")
            raise

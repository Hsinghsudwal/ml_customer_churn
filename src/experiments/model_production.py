from src.core.config_manager import ConfigManager


from mlflow.tracking import MlflowClient
import mlflow

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# mlflow.set_tracking_uri("http://127.0.0.1:8080")
# mlflow.set_experiment(EXPERIMENT_NAME)


def model_production(results, config_file):
    try:
        config = ConfigManager.load_file(config_file)
        mlflow_tracking_uri = config.get("mlflow_config", {}).get("mlflow_tracking_uri")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        model_name = results.get("best_model_name")
        client = MlflowClient()

        # Get the latest model in the Staging stage
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])

        if not staging_versions:
            logging.error(f"No staging model versions found for {model_name}")
            return "Staging model doesn't exist"

        staging_model = staging_versions[0]
        staging_version = staging_model.version
        logging.info(f"Latest Staging model version: {staging_version}")

        # Check if a Production version already exists
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])

        if prod_versions:
            # If production version exists, transition it to Archived
            current_production = prod_versions[0]
            production_version_number = current_production.version
            logging.info(
                f"Current Production model version: {production_version_number}"
            )

            # Transition the previous production model to archived
            client.transition_model_version_stage(
                name=model_name,
                version=production_version_number,
                stage="Archived",
                archive_existing_versions=False,
            )
            logging.info(
                "Previous Production version {production_version_number} ofv{model_name} has been archived"
            )

        # Transition the latest Staging model to Production
        client.transition_model_version_stage(
            name=model_name,
            version=staging_version,
            stage="Production",
            archive_existing_versions=False,
        )

        logging.info(
            f"Staging model version {staging_version} transitioned to Production"
        )

    except Exception as e:
        logging.error(f"Error occurred while staging model to production: {e}")
        raise

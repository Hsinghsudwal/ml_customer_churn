from mlflow.tracking import MlflowClient
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def model_staging(results):
    """Transitions a registered model to a new stage in MLflow"""
    try:
        stage = "Staging"

        model_name = results.get("best_model_name")
        client = MlflowClient()

        # Get the latest model version in "None" stage
        latest_versions = client.get_latest_versions(model_name, stages=["None"])

        if not latest_versions:
            raise ValueError(f"No versions found for model: {model_name}")

        # Pick the most recent model version
        model_version = latest_versions[0].version

        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage=stage,
            archive_existing_versions=True,
        )

        logging.info(
            f"Model {model_name} (Version {model_version}) moved to {stage} stage"
        )

    except Exception as e:
        logging.error(f"Error in model staging: {e}")
        raise

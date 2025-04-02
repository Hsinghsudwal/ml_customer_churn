import pandas as pd
import os
import logging
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
from core.config_artifact_store import ConfigManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# class ModelTracking:
#     def __init__(self, config_file):
#         self.config = ConfigManager.load_file(config_file)

def model_tracking(results, config_file):
        config = ConfigManager.load_file(config_file)

        logging.info("Model tracking and register")
        mlflow_tracking_uri = config.get("mlflow_config", {}).get(
            "mlflow_tracking_uri", {}
        )
        experiment_name = config.get("mlflow_config", {}).get("experiment_name", {})
        # model_name = config.get("mlflow_config", {}).get("model_name", "Ensemble Model")
    
        X_test = results.get("X_test")
        best_model = results.get("best_model")
        best_model_name = results.get("best_model_name", {})
        metrics = results.get("metrics", {})
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=experiment_name):
            mlflow.set_tag("Project", config.get("project_name", {}))
            mlflow.set_tag("Dev", config.get("author", {}))
            mlflow.log_metric("accuracy", metrics.get("accuracy", 0))
            mlflow.log_metric("precision", metrics.get("precision", 0))
            mlflow.log_metric("recall", metrics.get("recall", 0))
            mlflow.log_metric("f1_score", metrics.get("f1_score", 0))
            
            mlflow.log_artifact(__file__)
            # Log the model to MLflow
            signature = infer_signature(X_test, best_model.predict(X_test))
            mlflow.sklearn.log_model(
                best_model, artifact_path="model", signature=signature
            )
            # remote_server_uri=config.get("mlflow_config",{}).get("remote_server_uri",{})
            # mlflow.set_tracking_uri(remote_server_uri)
            # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            # # Model registry does not work with file store
            # if tracking_url_type_store != "file":
            #     mlflow.sklearn.log_model(
            #     best_model, "model", registered_model_name=best_model_name
            # )
            # else:
            #     mlflow.sklearn.log_model(best_model, "model")
            
            # register model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            result = mlflow.register_model(model_uri, best_model_name)
            print(
                f"Model registered with name {best_model_name} and version {result.version}"
            )
            logging.info("Model tracking and register completed")
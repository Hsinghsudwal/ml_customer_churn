import pandas as pd
import os
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report,
)
import logging
import json
import mlflow
from core.config_artifact_store import ConfigManager, ArtifactManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class ModelEvaluation:

    def __init__(self) -> None:
        pass

    # def save_metrics_to_json(self, metrics: Dict, output_path: str):
    #     # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #     with open(output_path, "w") as f:
    #         json.dump(metrics, f, indent=4)
    #     # logging.info(f"Metrics saved to {output_path}")

    def evaluation_metrics(self, y_test, y_pred, avg_method="binary"):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=avg_method)
        recall = recall_score(y_test, y_pred, average=avg_method)
        f1 = f1_score(y_test, y_pred, average=avg_method)
        cm = confusion_matrix(y_test, y_pred).tolist()
        classification_rep = classification_report(y_test, y_pred)

        return accuracy, precision, recall, f1, cm, classification_rep

    def model_evaluation(
        self, results, config: ConfigManager, artifact_store: ArtifactManager
    ):

        logging.info("Node 5: model evaluation")

        evaluate_dir = config.get("artifact_path", {}).get("evaluate", {})
        metrics_filename = config.get("artifact_path", {}).get("metrics_json", {})

        # Load existing evaluation if it exists
        metrics = artifact_store.load(evaluate_dir, metrics_filename)

        if metrics is not None:
            logging.info("Loaded evaluation from store. Skipping model evaluation.")
            return {"metrics": metrics}

        try:
            best_model = results.get("best_model")
            X_test = results.get("X_test")
            y_test = results.get("y_test")

            y_pred = best_model.predict(X_test)

            # Calculate metrics
            accuracy, precision, recall, f1, cm, classification_rep = (
                self.evaluation_metrics(y_test, y_pred)
            )

            metrics_json = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": cm,
                "classification_report": classification_rep,
            }

            # Create the full path to the metrics file
            metrics_path = os.path.join(evaluate_dir, metrics_filename)

            # Save metrics to a JSON file
            os.makedirs(evaluate_dir, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(metrics_json, f, indent=4)
            logging.info(f"Metrics saved to {metrics_path}")

            # Save evaluation as an artifact
            artifact_store.save(
                metrics_json, subdir=evaluate_dir, name=metrics_filename
            )

            return {"metrics": metrics_json}

        except Exception as e:
            logging.error(f"An error occurred during model evaluation: {e}")
            raise

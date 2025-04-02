import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import logging
import joblib
from typing import Dict, List
from core.config_artifact_store import ConfigManager, ArtifactManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class ModelTrainer:

    def __init__(self):
        pass

    def model_trainer(
        self, results: Dict, config: ConfigManager, artifact_store: ArtifactManager
    ):
        logging.info("Node 4: model training")

        # Check if the trained model already exists
        model_path_dir = config.get("artifact_path", {}).get("model_path", {})
        model_name = config.get("artifact_path", {}).get("saved_model", {})

        model = artifact_store.load(model_path_dir, model_name)

        if model is not None:
            logging.info("Loaded trained model from store. Skipping model training.")
            return {"best_model": model}

        X_train = results.get("X_train")
        X_test = results.get("X_test")
        y_train = results.get("y_train")
        y_test = results.get("y_test")

        # if val_train is None or val_test is None:
        #         logger.error(
        #             "Validation data not found in results. Transformation aborted."
        #         )
        #         raise ValueError("Missing validation data for transformation.")

        try:
            if isinstance(y_train, pd.DataFrame):
                y_train = y_train.squeeze()  # Convert DataFrame to Series
            if isinstance(y_test, pd.DataFrame):
                y_test = y_test.squeeze()
            # y_train and y_test are 1D arrays
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()

            # Define models
            models = {
                "Random Forest": RandomForestClassifier(
                    n_estimators=100, random_state=42
                ),
                "XGBoost": XGBClassifier(random_state=42),
                # "LightGBM": LGBMClassifier(random_state=42),
                # "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
                # "SVM": SVC(kernel='linear', random_state=42),
                "Tree": DecisionTreeClassifier(random_state=42),
            }

            model_results = []
            best_model = None
            best_model_name = ""
            best_accuracy = 0

            # Train individual models
            for name, model_instance in models.items():
                model_instance.fit(X_train, y_train)
                y_pred = model_instance.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                logging.info(f"Model: {name}, Accuracy: {accuracy * 100:.2f}%")
                model_results.append({"Model": name, "Accuracy": accuracy * 100})

                # Track the best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_instance
                    best_model_name = name

            # Train an ensemble model
            ensemble_model = VotingClassifier(
                estimators=[
                    ("rf", models["Random Forest"]),
                    ("xgb", models["XGBoost"]),
                    # ('lgbm', models["LightGBM"]),
                    # ('cat', models["CatBoost"]),
                    # ('svm', models["SVM"]),
                    ("tree", models["Tree"]),
                ],
                voting="hard",
            )

            ensemble_model.fit(X_train, y_train)
            y_pred = ensemble_model.predict(X_test)
            ensemble_accuracy = accuracy_score(y_test, y_pred)

            logging.info(f"Ensemble Model Accuracy: {ensemble_accuracy * 100:.2f}%")
            model_results.append(
                {"Model": "Ensemble Model", "Accuracy": ensemble_accuracy * 100}
            )

            # Check if the ensemble model is the best
            if ensemble_accuracy > best_accuracy:
                best_accuracy = ensemble_accuracy
                best_model = ensemble_model
                best_model_name = "Ensemble Model"

            logging.info(
                f"The best model is: {best_model} with accuracy: {best_accuracy * 100:.2f}%"
            )

            # Save the best model as an artifact
            artifact_store.save(best_model, subdir=model_path_dir, name=model_name)

            return {"best_model": best_model, "best_model_name": best_model_name}

        except Exception as e:
            logging.error(f"An error occurred during model training: {e}")
            raise

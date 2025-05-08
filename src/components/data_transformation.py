import logging
from typing import Dict, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from src.core.config_manager import ConfigManager
from src.core.artifact_manager import ArtifactManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataTransformation:

    def __init__(self):
        pass

    @staticmethod
    def remove_outliers(data: pd.DataFrame, labels: List[str]) -> pd.DataFrame:

        for label in labels:
            q1 = data[label].quantile(0.25)
            q3 = data[label].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            lower_bound = q1 - 1.5 * iqr

            data[label] = data[label].mask(
                data[label] < lower_bound, data[label].median()
            )
            data[label] = data[label].mask(
                data[label] > upper_bound, data[label].median()
            )
        return data

    def data_transformation(
        self,
        results,
        config: ConfigManager,
        artifact_store: ArtifactManager,
        pipeline_id,
    ):

        try:
            logging.info("Node 3: data transformer")

            # Retrieve file paths from configuration
            transform_path = config.get("artifact_path", {}).get("transform_path", {})
            transformer_filename = config.get("artifact_path", {}).get(
                "transformer", {}
            )
            xtrain_filename = config.get("artifact_path", {}).get("xtrain", {})
            xtest_filename = config.get("artifact_path", {}).get("xtest", {})
            ytrain_filename = config.get("artifact_path", {}).get("ytrain", {})
            ytest_filename = config.get("artifact_path", {}).get("ytest", {})

            # Check for existing transformation artifacts
            X_train_df = artifact_store.load(transform_path, xtrain_filename)
            X_test_df = artifact_store.load(transform_path, xtest_filename)
            y_train_df = artifact_store.load(transform_path, ytrain_filename)
            y_test_df = artifact_store.load(transform_path, ytest_filename)
            preprocessor = artifact_store.load(transform_path, transformer_filename)

            if all(
                item is not None
                for item in [X_train_df, X_test_df, y_train_df, y_test_df, preprocessor]
            ):
                logger.info("Transformation artifacts found. Skipping transformation.")
                return {
                    "X_train": X_train_df,
                    "X_test": X_test_df,
                    "y_train": y_train_df,
                    "y_test": y_test_df,
                    "preprocessor": preprocessor,
                }

            # Validate input data
            val_train = results.get("val_train_data")
            val_test = results.get("val_test_data")

            if val_train is None or val_test is None:
                logger.error(
                    "Validation data not found in results. Transformation aborted."
                )
                raise ValueError("Missing validation data for transformation.")

            # Remove outliers from numeric columns
            numeric_cols = val_train.select_dtypes(include=["int64", "float64"]).columns
            val_train = self.remove_outliers(val_train, numeric_cols)
            val_test = self.remove_outliers(val_test, numeric_cols)

            # Prepare features and target
            drop_cols = (
                ["Churn", "CustomerID"]
                if "CustomerID" in val_train.columns
                else ["Churn"]
            )
            X_train = val_train.drop(columns=drop_cols, errors="ignore")
            X_test = val_test.drop(columns=drop_cols, errors="ignore")

            y_train = val_train["Churn"].fillna(val_train["Churn"].mode()[0])
            y_test = val_test["Churn"].fillna(val_test["Churn"].mode()[0])

            # Identify features
            numeric_features = X_train.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()
            categorical_features = X_train.select_dtypes(
                include=["object"]
            ).columns.tolist()

            # Define transformation pipelines
            num_transformer = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_transformer = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="missing"),
                    ),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            # Create a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_transformer, numeric_features),
                    ("cat", cat_transformer, categorical_features),
                ],
                remainder="passthrough",
            )

            # Fit and transform data
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            # Get transformed column names
            cat_columns = (
                preprocessor.named_transformers_["cat"]
                .named_steps["encoder"]
                .get_feature_names_out(categorical_features)
            )
            new_columns = list(numeric_features) + list(cat_columns)

            # Convert transformed data into DataFrame
            X_train_df = pd.DataFrame(
                X_train_processed, columns=new_columns, index=X_train.index
            )
            X_test_df = pd.DataFrame(
                X_test_processed, columns=new_columns, index=X_test.index
            )

            # Convert target to DataFrame
            y_train_df = pd.DataFrame(y_train, columns=["Churn"])
            y_test_df = pd.DataFrame(y_test, columns=["Churn"])

            # Save transformation artifacts
            artifact_store.save(
                X_train_df,
                subdir=transform_path,
                name=xtrain_filename,
                pipeline_id=pipeline_id,
            )
            artifact_store.save(
                X_test_df,
                subdir=transform_path,
                name=xtest_filename,
                pipeline_id=pipeline_id,
            )
            artifact_store.save(
                y_train_df,
                subdir=transform_path,
                name=ytrain_filename,
                pipeline_id=pipeline_id,
            )
            artifact_store.save(
                y_test_df,
                subdir=transform_path,
                name=ytest_filename,
                pipeline_id=pipeline_id,
            )
            artifact_store.save(
                preprocessor,
                subdir=transform_path,
                name=transformer_filename,
                pipeline_id=pipeline_id,
            )

            logger.info("Data transformation completed.")

            return {
                "X_train": X_train_df,
                "X_test": X_test_df,
                "y_train": y_train_df,
                "y_test": y_test_df,
                "preprocessor": preprocessor,
            }

        except Exception as e:
            logger.error(f"Error during data transformation: {e}")
            raise

import os
import io
import json
import pickle
import logging
import pandas as pd

from src.core.artifact_manager import ArtifactManager 
from src.core.config_manager import ConfigManager

import boto3
from botocore.exceptions import NoCredentialsError


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LocalArtifactStore(ArtifactManager):
    """Local filesystem implementation of ArtifactStore."""

    def __init__(self, config: ConfigManager):
        """Initialize with a ConfigManager instance."""
        self.config = config

        # Get base artifact path from config
        self.base_path = self.config.get("artifact_path", {}).get("local", "artifacts")
        os.makedirs(self.base_path, exist_ok=True)
        logger.info(f"LocalArtifactStore initialized with base path: {self.base_path}")

    def save(self, artifact, subdir: str, name: str, pipeline_id: str = None) -> str:
        """Saves artifact locally with appropriate serialization based on file extension."""
        artifact_dir = os.path.join(self.base_path, subdir)
        os.makedirs(artifact_dir, exist_ok=True)
        artifact_path = os.path.join(artifact_dir, name)

        try:
            if name.endswith(".pkl"):
                with open(artifact_path, "wb") as f:
                    pickle.dump(artifact, f)
            elif name.endswith(".csv"):
                if isinstance(artifact, pd.DataFrame):
                    artifact.to_csv(artifact_path, index=False)
                else:
                    raise ValueError("CSV format only supports pandas DataFrames.")
            elif name.endswith(".json"):
                with open(artifact_path, "w") as f:
                    json.dump(artifact, f, indent=4)
            elif name.endswith(".txt"):
                with open(artifact_path, "w") as f:
                    f.write(str(artifact))
            else:
                # Default to pickle for unknown formats
                with open(artifact_path, "wb") as f:
                    pickle.dump(artifact, f)

            logger.info(f"Artifact '{name}' saved to {artifact_path}")
            
            # Handle metadata recording if implemented
            if pipeline_id and hasattr(self, 'metadata_store'):
                artifact_type = name.split(".")[-1]
                self.metadata_store.record_artifact(
                    pipeline_id, name, artifact_type, artifact_path
                )
                
            return artifact_path
        except Exception as e:
            logger.error(f"Error saving artifact {name}: {e}")
            raise

    def load(self, subdir: str, name: str):
        """Load an artifact in the specified format based on file extension."""
        artifact_path = os.path.join(self.base_path, subdir, name)
        try:
            if os.path.exists(artifact_path):
                if name.endswith(".pkl"):
                    with open(artifact_path, "rb") as f:
                        artifact = pickle.load(f)
                elif name.endswith(".csv"):
                    artifact = pd.read_csv(artifact_path)
                elif name.endswith(".json"):
                    with open(artifact_path, "r") as f:
                        artifact = json.load(f)
                elif name.endswith(".txt"):
                    with open(artifact_path, "r") as f:
                        artifact = f.read()
                else:
                    # Default to pickle for unknown formats
                    with open(artifact_path, "rb") as f:
                        artifact = pickle.load(f)

                logger.info(f"Artifact '{name}' loaded from {artifact_path}")
                return artifact
            else:
                logger.warning(f"Artifact '{name}' not found in {artifact_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading artifact {name}: {e}")
            raise


class CloudArtifactStore(ArtifactManager):
    """Cloud storage implementation of ArtifactStore (using AWS S3)."""


    def __init__(self, config: ConfigManager):
        """Initialize with a ConfigManager instance."""
        self.config = config
        self.bucket_name = (
            self.config.get("artifact_path", {}).get("cloud", {}).get("s3_bucket_name")
        )
        
        # AWS credentials from config
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=self.config.get("aws_access_key_id"),
            aws_secret_access_key=self.config.get("aws_secret_access_key"),
            region_name=self.config.get("aws_region"),
        )
        
        if not self.bucket_name:
            raise ValueError(
                "S3 bucket name must be specified in the configuration for cloud storage."
            )
        logger.info(f"CloudArtifactStore initialized with bucket: {self.bucket_name}")
        
        # Initialize metadata store if available
        if hasattr(self.config, "metadata_store"):
            self.metadata_store = self.config.metadata_store(config, self)



    def save(self, artifact, subdir: str, name: str, pipeline_id: str = None) -> str:
        """Saves artifact to AWS S3."""
        key = os.path.join(subdir, name)
        artifact_type = name.split(".")[-1]

        try:
            if name.endswith(".pkl"):
                buffer = pickle.dumps(artifact)
                self.s3_client.put_object(Bucket=self.bucket_name, Key=key, Body=buffer)
            elif name.endswith(".csv"):
                if isinstance(artifact, pd.DataFrame):
                    csv_buffer = artifact.to_csv(index=False).encode("utf-8")
                    self.s3_client.put_object(
                        Bucket=self.bucket_name, Key=key, Body=csv_buffer
                    )
                else:
                    raise ValueError("CSV format only supports pandas DataFrames.")
            elif name.endswith(".json"):
                json_buffer = json.dumps(artifact, indent=4).encode("utf-8")
                self.s3_client.put_object(
                    Bucket=self.bucket_name, Key=key, Body=json_buffer
                )
            elif name.endswith(".txt"):
                text_buffer = str(artifact).encode("utf-8")
                self.s3_client.put_object(
                    Bucket=self.bucket_name, Key=key, Body=text_buffer
                )
            else:
                buffer = pickle.dumps(artifact)
                self.s3_client.put_object(Bucket=self.bucket_name, Key=key, Body=buffer)

            s3_path = f"s3://{self.bucket_name}/{key}"
            logger.info(f"Artifact '{name}' saved to {s3_path}")
            
            # Record in metadata store if available and pipeline_id is provided
            if pipeline_id and hasattr(self, 'metadata_store'):
                self.metadata_store.record_artifact(
                    pipeline_id, name, artifact_type, s3_path
                )
                
            return s3_path
        except NoCredentialsError:
            logger.error("AWS credentials not found.")
            raise
        except Exception as e:
            logger.error(f"Error saving artifact {name} to S3: {e}")
            raise

    def load(self, subdir: str, name: str):
        """Load an artifact from AWS S3."""
        key = os.path.join(subdir, name)
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            file_content = response["Body"].read()

            if name.endswith(".pkl"):
                artifact = pickle.loads(file_content)
            elif name.endswith(".csv"):
                artifact = pd.read_csv(io.BytesIO(file_content))
            elif name.endswith(".json"):
                artifact = json.loads(file_content.decode("utf-8"))
            elif name.endswith(".txt"):
                artifact = file_content.decode("utf-8")
            else:
                artifact = pickle.loads(file_content)

            logger.info(f"Artifact '{name}' loaded from s3://{self.bucket_name}/{key}")
            return artifact
        except Exception as e:
            logger.error(f"Error loading artifact {name} from S3: {e}")
            return None


class LocalStackArtifactStore(CloudArtifactStore):
    """LocalStack implementation of ArtifactStore (using AWS S3 locally)."""

    def __init__(self, config: ConfigManager):
        """Initialize LocalStack S3 storage."""
        self.config = config
        self.bucket_name = (
            self.config.get("artifact_path", {})
            .get("localstack", {})
            .get("s3_bucket_name")
        )
        self.endpoint_url = (
            self.config.get("artifact_path", {})
            .get("localstack", {})
            .get("endpoint_url", "http://localhost:4566")
        )
        
        # LocalStack uses dummy credentials
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            endpoint_url=self.endpoint_url,
            region_name=self.config.get("aws_region", "us-east-1"),
        )
        
        if not self.bucket_name:
            raise ValueError(
                "S3 bucket name must be specified in the configuration for LocalStack."
            )
        logger.info(
            f"LocalStackArtifactStore initialized with bucket: {self.bucket_name} at {self.endpoint_url}"
        )
        
        # Initialize metadata store if available
        if hasattr(self.config, "metadata_store"):
            self.metadata_store = self.config.metadata_store(config, self)
    
    # Inherits save and load methods from CloudArtifactStore


class ArtifactStoreFactory:
    """Factory class for creating appropriate ArtifactStore instances."""

    @staticmethod
    def create_store(config: ConfigManager):
        """Create and return an appropriate artifact store based on configuration.
        
        Args:
            config: Configuration manager instance
            
        Returns:
            ArtifactManager: An instance of the appropriate artifact store
        """
        storage_type = config.get("storage_type", "local").lower()
        
        if storage_type == "local":
            return LocalArtifactStore(config)
        elif storage_type == "cloud":
            return CloudArtifactStore(config)
        elif storage_type == "localstack":
            return LocalStackArtifactStore(config)
        else:
            raise ValueError(
                f"Unsupported storage type: {storage_type}. Must be 'local', 'cloud', or 'localstack'."
            )


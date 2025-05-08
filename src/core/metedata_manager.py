
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MetadataStore:
    """Store metadata about artifacts."""
    
    def __init__(self, config, artifact_store):
        self.config = config
        self.artifact_store = artifact_store
        
    def record_artifact(self, pipeline_id, name, artifact_type, path):
        """Record metadata about an artifact."""
        logger.info(f"Recording metadata for artifact '{name}' from pipeline '{pipeline_id}'")
        # Implementation details would go here
        
        
        
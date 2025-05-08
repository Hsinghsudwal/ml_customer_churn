import logging
from .config_manager import ConfigManager
from .artifact_manager import ArtifactManager

logger = logging.getLogger(__name__)


class Mlstack:
    def __init__(
        self, name, config: ConfigManager, nodes, artifact_manager: ArtifactManager
    ):
        self.name = name
        self.config = config
        self.nodes = nodes
        self.artifact_manager = artifact_manager

    def run(self, pipeline_id: str):
        results = {}

        # Run all nodes sequentially
        for node in self.nodes:
            try:
                output = node(
                    results, self.config, self.artifact_manager, pipeline_id=pipeline_id
                )
                results.update(output)
                # logger.info(f"Node {node.__name__} completed successfully.")
            except Exception as e:
                logging.error(f"Node {node.__name__} failed with error: {e}")
                # Early exit if any critical node fails
                return results

        # logger.info(f"Completed Mlstack '{self.name}' execution")
        return results

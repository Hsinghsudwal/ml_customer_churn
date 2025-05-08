from prefect.deployments import Deployment
from prefect.infrastructure import Process
from prefect.server.schemas.schedules import CronSchedule  # Optional: for scheduling
from ml_pipeline import my_ml_pipeline  # import your flow here

if __name__ == "__main__":
    Deployment.build_from_flow(
        flow=my_ml_pipeline,
        name="ml-pipeline-deployment",
        version="1.0.0",
        work_queue_name="default",
        infrastructure=Process(),  # can swap with DockerContainer(), KubernetesJob(), etc.
        tags=["ml", "metadata", "versioning"],
        description="A metadata-aware ML pipeline with artifact tracking.",
        parameters={},  # Add default parameters here if your flow accepts any
        schedule=None,  # Or use: CronSchedule(cron="0 12 * * *") for daily noon runs
    ).apply()

import argparse
from pipelines.training_pipeline import TrainingPipeline
from pipelines.experiment_pipeline import ExperimentPipeline

# import logging

# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )


def main():
    # Set up parser
    parser = argparse.ArgumentParser(description="Running the pipelines")

    parser.add_argument("--local", action="store_true", help="Use local file storage")
    parser.add_argument("--cloud", action="store_true", help="Use AWS cloud storage")
    parser.add_argument(
        "--localstack", action="store_true", help="Use LocalStack for testing"
    )
    parser.add_argument(
        "--data", default="data/churn-train.csv", help="Path to dataset"
    )
    # parser.add_argument("--localstack", action="store_true", help="Use LocalStack for testing")

    # Parse arguments
    args = parser.parse_args()

    if args.local:
        config_file = "config/local.yaml"
    elif args.cloud:
        config_file = "config/cloud.yaml"
    elif args.localstack:
        config_file = "config/localstack.yaml"
    else:
        print("Please define config file")  # Default to local

    # Initialize pipeline
    pipeline = TrainingPipeline(
        data_path=args.data,
        config_file=config_file,
    )

    # Run pipeline
    results = pipeline.run()

    ExperimentPipeline(config_file).run(results, config_file)


if __name__ == "__main__":
    main()

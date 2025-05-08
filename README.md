# ML Customer Churn Prediction Pipeline

A complete machine learning pipeline for predicting customer churn using ensemble methods.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Installation](#installation)
- [Project Setup](#project-setup)
- [Notebooks](#notebooks)
- [Pipeline Components](#pipeline-components)
- [Experiment Tracking](#experiment-tracking)
- [Model Performance](#model-performance)
- [Deployment](#deployment)
- [Next Steps](#next-steps)
- [Docker Integration](#docker-integration)

## Problem Statement
Customer retention is critical for subscription-based businesses such as telecom, banking. Retaining existing customers is often more cost-effective than acquiring new ones. A key metric in this domain is customer churn — when users discontinue service. Proactively identifying customers likely to churn allows for targeted interventions, improving retention rates and reducing revenue loss.

Objective
The goal is to build a robust, end-to-end Machine Learning pipeline that predicts customer churn based on historical data. The system should:

- Ingest and validate raw customer data.

- Preprocess and transform features for modeling.

- Train multiple models and select the best one based on accuracy.

- Evaluate the model and log performance metrics.

- Save all artifacts for reproducibility and future deployment.

## Installation

```bash
# Clone the repository
git clone https://github.com/Hsinghsudwal/ml_customer_churn.git
cd customer-churn-prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Setup

1. Ensure your data file is in the proper location:
   - Training data should be placed in `data/churn-train.csv`

2. Configure pipeline parameters in `config.yaml` (if applicable)

3. Run the pipeline:
   ```bash
   python run.py (--local, --cloud, --localstack)
   ```

## Notebooks

The project includes Jupyter notebooks for:

1. **Exploratory Data Analysis**: `notebooks/notebook.ipynb`
   - Initial data exploration
   - Feature importance analysis
   - Correlation studies

2. **Model Evaluation**: `notebooks/notebook.ipynb`
   - Detailed model performance evaluation
   - Confusion matrices
   - ROC curves and precision-recall curves


## Pipeline Components

The pipeline consists of 5 main components that run sequentially:

### 1. Data Ingestion
- Loads data from source CSV file
- Splits into training and test sets
- Saves processed datasets to `outputs_store/raw/`

### 2. Data Validation
- Validates data for correctness, completeness, and consistency
- Checks for missing values, outliers, and data types
- Produces validated datasets in `outputs_store/validate/`

### 3. Data Transformation
- Applies feature engineering and preprocessing
- Handles categorical variables, scaling, and encoding
- Creates feature matrices (X) and target vectors (y)
- Saves preprocessed data and transformer model to `outputs_store/transformer/`

### 4. Model Training
- Trains multiple model candidates:
  - Random Forest
  - XGBoost
  - Decision Tree
- Creates an ensemble voting classifier
- Selects best performing model
- Saves model artifacts to `outputs_store/model/`

### 5. Model Evaluation
- Evaluates model on test data
- Calculates metrics (accuracy, precision, recall, F1)
- Saves evaluation metrics to `outputs_store/evaluate/`


## Experiment Tracking
mlflow: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts

  * Model tracking and registry
  * Model Staging
  * Model Production

Experiments are tracked through detailed logging and artifact storage:

- **Logging**: All pipeline steps generate logs with INFO and WARNING level messages
- **Artifact Storage**: All artifacts are stored in `outputs/_store/` directory structure
- **Pipeline Execution**: Each pipeline run generates a unique ID (e.g., `52c9263c-a5d7-41b9-9183-771abc0b1d04`)
- **Metrics**: Final model evaluation metrics are stored in JSON format


### Log Example
```
2025-05-08 10:53:24,671 - INFO - Starting pipeline 'Churn Training Pipeline' with ID: 52c9263c-a5d7-41b9-9183-771abc0b1d04
2025-05-08 10:57:54,331 - INFO - Pipeline completed successfully in 0:04:29.659749
```

## Model Performance

Based on the latest run, the following models were evaluated:

| Model | Accuracy |
|-------|----------|
| Random Forest | 99.94% |
| XGBoost | 99.99% |
| Decision Tree | 99.99% |
| **Ensemble (Voting)** | **100.00%** |

The ensemble voting classifier combining all three models achieved the best performance and was selected as the final model.

### Deployment

To deploy the model:

1. **Model Serving**:
   - Export the model using the provided script:
     ```bash
     python src/deployment/app.py
     ```
   - Deploy as a REST API using Flask

2. **Batch Prediction**:
   - Use the batch prediction script:
     ```bash
     python src/deployment/app.py --input data/new_customers.csv --output predictions/results.csv
     ```

## Next Steps

### Monitoring

Monitor your deployed model with:

1. **Data Drift Detection**:
   - Set up a monitoring service using `src/monitoring/drift_detector.py`
   - Configure alerts for significant distribution changes

2. **Model Performance Tracking**:
   - Track accuracy metrics over time
   - Set up retraining triggers when performance drops below thresholds


### Improvements

1. **Feature Engineering**:
   - Experiment with additional features
   - Apply more advanced feature selection techniques

2. **Model Improvements**:
   - Test more sophisticated models (Neural Networks, SVM)
   - Fine-tune hyperparameters with more extensive grid search

3. **Deployment Optimization**:
   - Containerize the application
   - Set up CI/CD pipeline for model updates

4. **A/B Testing**:
   - Implement A/B testing framework to compare model versions in production

## Docker Integration

A Dockerfile is provided to containerize the application:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the pipeline
CMD ["python", "run.py"]
```



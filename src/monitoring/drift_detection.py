import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset

# Load datasets
reference_data = pd.read_csv("outputs_store/raw/train.csv")
current_data = pd.read_csv("data/live_batch.csv")

# Drop target column if present (drift is usually calculated on features only)
target_col = "Churn"
if target_col in reference_data.columns:
    reference_data = reference_data.drop(columns=[target_col])
if target_col in current_data.columns:
    current_data = current_data.drop(columns=[target_col])

# Create Evidently Report for Data Drift
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)

# Generate and save HTML report
report.save_html("reports/data_drift_report.html")
print("✅ Drift report saved to 'reports/data_drift_report.html'")

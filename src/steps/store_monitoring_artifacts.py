import os
import mlflow
from zenml import step

@step
def store_monitoring_artifacts(report_path: str) -> None:
    """
    Logs the evidently drift report HTML to the MLflow tracking server.
    
    Args:
        report_path: The path to the JSON report. We will find the corresponding HTML report next to it.
    """
    html_path = report_path.replace(".json", ".html")
    
    if os.path.exists(html_path):
        # The ZenML @pipeline context automatically handles the MLflow experiment/run initialization
        # if the mlflow_tracker is configured.
        print(f"Logging {html_path} to MLflow artifacts...")
        mlflow.log_artifact(html_path, artifact_path="evidently_drift_reports")
    else:
        print(f"Warning: HTML drift report not found at {html_path}")

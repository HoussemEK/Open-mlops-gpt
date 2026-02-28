import json
from zenml import step
from zenml.client import Client

@step
def trigger_decision(report_path: str) -> bool:
    """
    Parses the Evidently drift report and decides if the model needs retraining.
    
    Args:
        report_path: Path to the JSON drift report.
        
    Returns:
        True if drift is detected (retraining needed), False otherwise.
    """
    with open(report_path, "r") as f:
        report = json.load(f)
        
    # Evidently JSON output structure is quite deep. The DataDriftPreset returns a top-level dataset_drift boolean
    # Let's safely extract it.
    drift_detected = False
    try:
        # Assuming the first metric block in the preset contains dataset drift status
        for metric in report.get("metrics", []):
            if metric.get("metric") == "DatasetDriftMetric":
                drift_detected = metric.get("result", {}).get("dataset_drift", False)
                break
    except Exception as e:
        print(f"Error parsing drift report: {e}")
        
    if drift_detected:
        print("🚨 SIGNIFICANT DATA DRIFT DETECTED! 🚨")
        print("Initiating retrain process...")
        # Note: In a fully automated setup with ZenML Cloud or an advanced Orchestrator,
        # you would trigger the `training_pipeline` here programmatically.
        # For simplicity in this local Phase 3 demo, we just return the boolean flag.
    else:
        print("✅ No data drift detected. Inference streams look healthy.")
        
    return drift_detected

from zenml import pipeline

from src.steps.collect_inference_data import collect_inference_data
from src.steps.run_evidently_report import run_evidently_report
from src.steps.trigger_decision import trigger_decision
from src.steps.store_monitoring_artifacts import store_monitoring_artifacts

@pipeline(enable_cache=False)
def monitoring_pipeline(test_data_path: str, simulate_drift: bool = False):
    """
    Phase 3: The Monitoring Pipeline.
    
    1. Collects inference data (simulating clean or drifted traffic).
    2. Runs Evidently to compute data/target drift over generic image representations.
    3. Saves the report and logs the visual HTML to MLflow.
    4. Triggers a decision on whether to retrain based on the drift threshold.
    """
    # 1. Collect Data
    reference_df, current_df = collect_inference_data(
        test_data_path=test_data_path,
        simulate_drift=simulate_drift
    )
    
    # 2. Run Report
    report_path = run_evidently_report(
        reference_df=reference_df, 
        current_df=current_df
    )
    
    # 3. Store Artifacts
    store_monitoring_artifacts(report_path=report_path)
    
    # 4. Trigger Retrain Decision
    retrain_flag = trigger_decision(report_path=report_path)
    
    return retrain_flag

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    # If the user passes '--drifted' in the Docker command, we run the drifted scenario
    parser.add_argument("--drifted", action="store_true", help="Simulate data drift")
    args = parser.parse_args()
    
    dataset_path = "data/processed/test.npz"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Please run Training Pipeline first.")
        exit(1)
        
    print(f"Starting OpenMLOps Monitoring Pipeline ... (Simulate Drift={args.drifted})")
    
    monitoring_pipeline(
        test_data_path=dataset_path, 
        simulate_drift=args.drifted
    )

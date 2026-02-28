import json
import os

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import pandas as pd
from zenml import step

@step
def run_evidently_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    report_path: str = "reports/drift_report.json"
) -> str:
    """
    Computes an Evidently data drift report on the tabular image features.
    
    Args:
        reference_df: Baseline data features (e.g. from the test set without drift).
        current_df: Current inference features (potentially drifted).
        report_path: The file path to save the output JSON report.
        
    Returns:
        The path where the drift report JSON was saved.
    """
    # Create the reports folder if it doesn't exist (when running locally)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # 1. Initialize the Evidently Report with presets
    report = Report(metrics=[
        DataDriftPreset(), 
        TargetDriftPreset() # We use 'prediction' columns as targets
    ])
    
    # We specify column mapping to explicitly tell Evidently what everything is
    from evidently.pipeline.column_mapping import ColumnMapping
    column_mapping = ColumnMapping()
    column_mapping.target = "target"
    if "prediction" in current_df.columns:
        column_mapping.prediction = "prediction"
        
    # The reference df needs a prediction column for TargetDriftPreset, just use target
    if "prediction" not in reference_df.columns:
        reference_df["prediction"] = reference_df["target"]
        
    if "target" not in current_df.columns:
        # We might not know true target in current inference data, mock it for the Preset safely
        current_df["target"] = current_df["prediction"]

    # 2. Run computation
    report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)
    
    # 3. Save as JSON
    report.save_json(report_path)
    
    # Also save an HTML copy for human viewing later
    html_path = report_path.replace(".json", ".html")
    report.save_html(html_path)
    
    return report_path

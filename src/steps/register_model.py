"""
Step: register_model
====================
Registers the trained CNN into the MLflow Model Registry,
tagging it with the run ID and assigning it to the 'Staging' alias/stage.

Satisfies:
  R-TRAIN-7 — Register model in MLflow Model Registry
  R-MODEL-5 — Model registered with versioning, stage -> Staging
"""
import logging
import mlflow
import torch.nn as nn
from zenml import step
from zenml.client import Client

logger = logging.getLogger(__name__)

@step(experiment_tracker="mlflow_tracker")
def register_model(
    model: nn.Module,
    metrics: dict,
    model_name: str = "cifar10-cnn"
) -> None:
    """
    Logs the model artifact to MLflow and registers it in the registry.
    """
    # Grab current active run info
    run = mlflow.active_run()
    if not run:
        logger.warning("No active MLflow run found. Cannot register model.")
        return
        
    run_id = run.info.run_id
    logger.info("Logging pyfunc model to MLflow for run %s ...", run_id)
    
    # Log the PyTorch model
    # Note: we log it inside the active run first
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name=model_name
    )
    
    logger.info("Model registered in MLflow registry as '%s'.", model_name)
    
    # In newer MLflow versions (>=2.9), the preferred way to transition is using Model Aliases.
    # Older MLflow uses "Stages" (Staging, Production... etc).
    # We will use the traditional client stage transition for compatibility.
    client = mlflow.MlflowClient()
    
    # Get the latest version of the model we just registered
    latest_versions = client.get_latest_versions(name=model_name, stages=["None"])
    if latest_versions:
        latest_version = latest_versions[0].version
        logger.info("Transitioning model '%s' version %s to Staging", model_name, latest_version)
        
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Staging",
            archive_existing_versions=True
        )
        
        # Add tags
        client.set_model_version_tag(model_name, latest_version, "run_id", run_id)
        if "test_accuracy" in metrics:
            acc = f"{metrics['test_accuracy']:.4f}"
            client.set_model_version_tag(model_name, latest_version, "test_accuracy", acc)
            
        logger.info("✅ Model registered successfully to Staging.")

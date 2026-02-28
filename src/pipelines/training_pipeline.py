"""
Training Pipeline
=================
Wires all 8 ZenML steps together.

Satisfies:
  R-TRAIN-1 through R-TRAIN-8 (all steps)
  R-MODEL-1 through R-MODEL-5
"""
import logging
import os
from zenml import pipeline
from zenml.client import Client

from src.steps.ingest_data import ingest_data
from src.steps.validate_data import validate_data
from src.steps.split_data import split_data
from src.steps.preprocess import preprocess
from src.steps.train import train_model
from src.steps.evaluate import evaluate_model
from src.steps.register_model import register_model
from src.steps.export_model import export_model

logger = logging.getLogger(__name__)

# ── Pipeline Definition ──────────────────────────────────────────────

@pipeline(enable_cache=False)
def training_pipeline(
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 0.001,
    dropout_rate: float = 0.5
) -> None:
    """
    Main Training Pipeline for OpenMLOps Challenge.
    Executes: ingest -> validate -> preprocess -> split -> train -> eval -> register -> export.
    """
    # ── Phase 1: Data Layer ──
    raw_data_path = ingest_data()
    validated_raw_dir = validate_data(raw_data_path=raw_data_path)
    processed_dir = split_data(raw_data_path=validated_raw_dir)
    processed_files = preprocess(processed_data_path=processed_dir)
    
    # ── Phase 2: Model & Training ──
    trained_model = train_model(
        processed_data_path=processed_files,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        dropout_rate=dropout_rate
    )
    
    metrics = evaluate_model(
        model=trained_model,
        processed_data_path=processed_files,
        batch_size=batch_size
    )
    
    # ── Model Registry & Export ──
    register_model(
        model=trained_model,
        metrics=metrics
    )
    
    export_model(model=trained_model)


# ─────────────────────────────────────────────────────────────────
# Entry point for `docker compose run training-runner`
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting OpenMLOps Training Pipeline ...")
    
    # We must ensure the MLflow tracker is available in the active stack
    # The setup of the stack occurs manually or via bash script, but we
    # run the pipeline directly here when invoked.
    
    try:
        training_pipeline(
            epochs=15,
            batch_size=128,
            lr=0.001,
            dropout_rate=0.5
        )
        logger.info("✅ Training Pipeline completed successfully!")
    except Exception as e:
        logger.exception("❌ Pipeline failed: %s", e)
        raise

from typing import Dict, List

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)


def get_trainer(
    earlystop: Dict,
    checkpoints: List[Dict],
    max_epochs: int,
    num_sanity_val_steps: int,
    accumulate_batch: int,
):
    """
    Create a PyTorch Lightning module for segmentation with
    the given model and optimization configuration.

    Args:
        parameters for optimization.
        model: The PyTorch model to use for segmentation.

    Returns:
        trainer: return a trainer object
    """

    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop_callback = EarlyStopping(**earlystop)
    list_callbacks = [ModelCheckpoint(**checkpoint) for checkpoint in checkpoints]

    list_callbacks.append(early_stop_callback)
    list_callbacks.append(lr_monitor)

    strategy = "auto"

    trainer = pl.Trainer(
        callbacks=list_callbacks,
        max_epochs=max_epochs,
        num_sanity_val_steps=num_sanity_val_steps,
        strategy=strategy,
        log_every_n_steps=2,
        accumulate_grad_batches=accumulate_batch,
    )

    return trainer

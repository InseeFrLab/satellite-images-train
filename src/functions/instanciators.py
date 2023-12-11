from typing import Dict, List

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from config.loss import loss_dict
from config.module import module_dict
from config.task import task_dict
from optim.optimizer import generate_optimization_elements


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


def get_model(module_name: str, n_channel: int):
    """
    Instantiate a module based on the provided module type.

    Args:
        module_type (str): Type of module to instantiate.

    Returns:
        object: Instance of the specified module.
    """
    if module_name not in module_dict:
        raise ValueError("Invalid module type")

    return module_dict[module_name](n_channel)


def get_loss(loss_name: str):
    """
    intantiates an optimizer object with the parameters
    specified in the configuration file.

    Args:
        model: A PyTorch model object.
        config: A dictionary object containing the configuration parameters.

    Returns:
        An optimizer object from the `torch.optim` module.
    """

    if loss_name not in loss_dict:
        raise ValueError("Invalid loss type")
    else:
        return loss_dict[loss_name]()


def get_lightning_module(
    module_name: str,
    loss_name: str,
    n_channel: int,
    task: str,
    lr: float,
    momentum: float,
    earlystop: Dict,
    scheduler_patience: int,
):
    """
    Create a PyTorch Lightning module for segmentation
    with the given model and optimization configuration.

    Args:
        config (dict): Dictionary containing the configuration
        parameters for optimization.
        model: The PyTorch model to use for segmentation.

    Returns:
        A PyTorch Lightning module for segmentation.
    """
    list_params = generate_optimization_elements(lr, momentum, earlystop, scheduler_patience)

    if task not in task_dict:
        raise ValueError("Invalid task type")
    else:
        LightningModule = task_dict[task]

    model = get_model(module_name, n_channel)
    loss = get_loss(loss_name)

    lightning_module = LightningModule(
        model=model(),
        loss=loss(),
        optimizer=list_params[0],
        optimizer_params=list_params[1],
        scheduler=list_params[2],
        scheduler_params=list_params[3],
        scheduler_interval=list_params[4],
    )

    return lightning_module

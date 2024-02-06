from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
from albumentations import Compose
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from s3fs import S3FileSystem

from config.dataset import dataset_dict
from config.loss import loss_dict
from config.module import module_dict
from config.task import task_dict
from config.scheduling import scheduling_policies


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


def get_dataset(
    task: str,
    patchs: List,
    labels: List,
    n_bands: int,
    fs: S3FileSystem,
    transform: Optional[Compose] = None,
):
    """
    intantiates a dataset given a task.

    Args:
        task: The considered task.

    Returns:
        A Dataset object.
    """

    if task not in dataset_dict:
        raise ValueError("Invalid dataset type")
    else:
        return dataset_dict[task](patchs, labels, n_bands, fs, transform)


def get_model(module_name: str, n_bands: str):
    """
    Instantiate a module based on the provided module type.

    Args:
        module_type (str): Type of module to instantiate.

    Returns:
        object: Instance of the specified module.
    """
    if module_name not in module_dict:
        raise ValueError("Invalid module type")

    return module_dict[module_name](n_bands)


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


def get_scheduler(scheduler_name: str):
    """
    Instantiates a scheduler from a policy.

    Args:
        scheduling_name (str): Policy.
    """
    if scheduler_name not in scheduling_policies:
        raise ValueError("Invalid scheduler.")
    else:
        return scheduling_policies[scheduler_name]


def get_lightning_module(
    module_name: str,
    loss_name: str,
    n_bands: str,
    task: str,
    lr: float,
    momentum: float,
    earlystop: Dict,
    scheduler_name: str,
    scheduler_patience: int,
    cuda: int,
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

    if task not in task_dict:
        raise ValueError("Invalid task type")
    else:
        LightningModule = task_dict[task]

    model = get_model(module_name, n_bands)
    if cuda:
        model.cuda()
    loss = get_loss(loss_name)

    # TODO: faire get_optimizer with kwargs
    # TODO: AdamW ?
    optimizer = torch.optim.Adam
    optimizer_params = {"lr": lr}

    # TODO: faire get_scheduler with kwargs
    scheduler = get_scheduler(scheduler_name)
    scheduler_params = {
        "monitor": "validation_loss",
        "mode": "min",
        "patience": scheduler_patience,
    }
    scheduler_interval = "epoch"

    lightning_module = LightningModule(
        model=model,
        loss=loss,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        scheduler=scheduler,
        scheduler_params=scheduler_params,
        scheduler_interval=scheduler_interval,
    )

    return lightning_module

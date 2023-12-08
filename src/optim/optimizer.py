from typing import Dict

import torch


def generate_optimization_elements(
    lr: float, momentum: float, earlystop: Dict, scheduler_patience: int
):
    """
    Returns the optimization elements required for PyTorch training.


    Returns:
        tuple: A tuple containing the optimizer, optimizer parameters,
        scheduler, scheduler parameters, and scheduler interval.

    """

    optimizer = torch.optim.SGD
    optimizer_params = {"lr": lr, "momentum": momentum}

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = {
        "monitor": earlystop["monitor"],
        "mode": earlystop["mode"],
        "patience": scheduler_patience,
    }  # TODO: vérifier si ok d'utilise config d'early stop ici.
    # IMPORTANT CAR PEUT ETRE CONFIG A REVOIR
    scheduler_interval = "epoch"

    return (
        optimizer,
        optimizer_params,
        scheduler,
        scheduler_params,
        scheduler_interval,
    )

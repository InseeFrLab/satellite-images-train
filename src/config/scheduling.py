"""
Scheduling policies
"""

import torch

scheduling_policies = {
    "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "one_cycle": torch.optim.lr_scheduler.OneCycleLR,
}

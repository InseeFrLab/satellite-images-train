from torch.nn import BCELoss, CrossEntropyLoss, BCEWithLogitsLoss
import torch


class WeightedCrossEntropyLoss(CrossEntropyLoss):
    def __new__(cls, building_class_weight: float):
        weight = torch.Tensor([1, building_class_weight])
        return CrossEntropyLoss(weight=weight)


class WeightedBCEWithLogitsLoss(BCEWithLogitsLoss):
    def __new__(cls, building_class_weight: float):
        pos_weight = torch.Tensor([building_class_weight])
        return BCEWithLogitsLoss(pos_weight=pos_weight)


loss_dict = {
    "crossentropy": {"loss_function": CrossEntropyLoss, "weighted": False, "kwargs": {}},
    "crossentropy_ignore_0": {
        "loss_function": CrossEntropyLoss,
        "weighted": False,
        "kwargs": {
            "ignore_index": 0,
        },
    },
    "crossentropy_weighted": {
        "loss_function": WeightedCrossEntropyLoss,
        "weighted": True,
        "kwargs": {},
    },
    "bce": {"loss_function": BCELoss, "weighted": False, "kwargs": {}},
    "bce_logits_weighted": {
        "loss_function": WeightedBCEWithLogitsLoss,
        "weighted": True,
        "kwargs": {},
    },
}

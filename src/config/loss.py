import torch
from torch import nn
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss


class WeightedCrossEntropyLoss(CrossEntropyLoss):
    def __new__(cls, weights: list):
        weight = torch.Tensor(weights)
        return CrossEntropyLoss(weight=weight)


class SmoothedBCEWithLogitsLoss(nn.Module):
    # TODO: add this to the pipeline.
    def __init__(
        self,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        pos_weight: torch.Tensor = torch.Tensor([1.0]),
    ):
        super(SmoothedBCEWithLogitsLoss, self).__init__()
        assert 0 <= label_smoothing < 1, "label_smoothing value must be between 0 and 1."
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.bce_with_logits = BCEWithLogitsLoss(reduction=reduction, pos_weight=pos_weight)

    def forward(self, input, target):
        # TODO: add LongTensor type check
        target = target.to(torch.float32)
        if self.label_smoothing > 0:
            positive_smoothed_labels = 1.0 - self.label_smoothing
            negative_smoothed_labels = self.label_smoothing
            target = target * positive_smoothed_labels + (1 - target) * negative_smoothed_labels

        loss = self.bce_with_logits(input, target)
        return loss


class WeightedBCEWithLogitsLoss(SmoothedBCEWithLogitsLoss):
    def __new__(cls, label_smoothing: float, weights: list):
        pos_weight = torch.Tensor(max(weights))
        return SmoothedBCEWithLogitsLoss(label_smoothing=label_smoothing, pos_weight=pos_weight)


class CustomBCELoss(nn.Module):
    """
    Custom BCELoss with target cast to float.
    Only works if input is probability !
    """

    def __init__(self):
        super(CustomBCELoss, self).__init__()
        self.loss = BCELoss()

    def forward(self, input, target):
        # TODO: add LongTensor type check
        target = target.to(torch.float32)
        return self.loss(input, target)


# TODO: add focal loss
loss_dict = {
    "cross_entropy": {
        "loss_function": CrossEntropyLoss,
        "weighted": False,
        "smoothing": False,
        "kwargs": {},
    },
    "cross_entropy_ignore_0": {
        "loss_function": CrossEntropyLoss,
        "weighted": False,
        "smoothing": False,
        "kwargs": {
            "ignore_index": 0,
        },
    },
    "cross_entropy_weighted": {
        "loss_function": WeightedCrossEntropyLoss,
        "weighted": True,
        "smoothing": False,
        "kwargs": {},
    },
    "bce": {"loss_function": CustomBCELoss, "weighted": False, "smoothing": False, "kwargs": {}},
    "bce_logits_weighted": {
        "loss_function": WeightedBCEWithLogitsLoss,
        "weighted": True,
        "smoothing": True,
        "kwargs": {},
    },
}

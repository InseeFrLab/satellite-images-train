from torch.nn import BCELoss, CrossEntropyLoss

loss_dict = {
    "crossentropy": CrossEntropyLoss,
    "bce": BCELoss,
}

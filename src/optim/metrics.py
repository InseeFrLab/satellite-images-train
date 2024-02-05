import torch


def IOU(output, labels):
    """
    Calculate Intersection Over Union indicator
    for the positive class of a segmentation task
    based on output segmentation mask of a model
    and the true segmentations mask.

    Args:
        output: Output of the segmentation model.
        label: True segmentation mask.

    """
    if output.dim() == 3:
        # Single class: if > 0.5, prediction is 1
        preds = (output > 0.5).float()
    else:
        preds = torch.argmax(output, axis=1)

    numIOU = torch.sum((preds * labels), axis=[1, 2])  # vaut 1 si les 2 = 1
    denomIOU = torch.sum(torch.clamp(preds + labels, max=1), axis=[1, 2])

    IOU = numIOU / denomIOU
    IOU = torch.tensor(
        [1 if torch.isnan(x) else x for x in IOU], dtype=torch.float
    )  # TODO: ici si on a des images sans aucun pixel bâtiment
    # on "biaise" potentiellement l'IOU vers le haut, parce que si
    # on prédit rien on a une IOU de 1. N'arrive pas si on
    # n'a pas d'images sans pixel bâtiment

    return torch.mean(IOU)

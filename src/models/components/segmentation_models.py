import torchvision
from torch import nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class DeepLabv3Module(nn.Module):
    """
    From the paper https://arxiv.org/abs/1706.05587
    segmentation model using atrous convolution to
    take into account multiscales effect.
    """

    def __init__(self, n_bands="3"):
        """
        Module constructor.

        Args:
            n_bands (str): Number of channels of the input image.
        """
        super().__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(
            weights="DeepLabV3_ResNet101_Weights.DEFAULT"
        )
        # 1 classe !
        self.model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))

        if n_bands != "3":
            self.model.backbone["conv1"] = nn.Conv2d(
                int(n_bands),
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )

    def forward(self, x):
        """
        Forward method.
        """
        return self.model(x)["out"]


class SingleClassDeepLabv3Module(nn.Module):
    """
    From the paper https://arxiv.org/abs/1706.05587
    segmentation model using atrous convolution to
    take into account multiscales effect.
    """

    def __init__(self, n_bands: int = "3"):
        """
        Module constructor.

        Args:
            n_bands (str): Number of channels of the input image.
        """
        super().__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(
            weights="DeepLabV3_ResNet101_Weights.DEFAULT"
        )
        # 1 classe !
        self.model.classifier = DeepLabHead(2048, 1)

        if n_bands != "3":
            self.model.backbone["conv1"] = nn.Conv2d(
                int(n_bands),
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )

    def forward(self, x):
        """
        Forward method.
        """
        return self.model(x)["out"].squeeze()

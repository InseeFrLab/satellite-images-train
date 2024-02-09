import torchvision
from torch import nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from transformers import SegformerForSemanticSegmentation


class DeepLabv3Module(nn.Module):
    """
    From the paper https://arxiv.org/abs/1706.05587
    segmentation model using atrous convolution to
    take into account multiscales effect.
    """

    def __init__(self, n_bands: int = 3, logits: bool = True):
        """
        Module constructor.

        Args:
            n_bands (int): Number of channels of the input image.
            logits (bool): True if logits out, if False probabilities.
        """
        super().__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(
            weights="DeepLabV3_ResNet101_Weights.DEFAULT"
        )
        # 1 classe !
        self.model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
        self.softmax_layer = nn.Softmax(dim=1)
        self.logits = logits

        if n_bands != 3:
            self.model.backbone["conv1"] = nn.Conv2d(
                n_bands,
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
        logits = self.model(x)["out"]
        if self.logits:
            return logits
        else:
            return self.softmax_layer(logits)


class SingleClassDeepLabv3Module(nn.Module):
    """
    From the paper https://arxiv.org/abs/1706.05587
    segmentation model using atrous convolution to
    take into account multiscales effect.
    """

    def __init__(self, n_bands: int = 3, logits: bool = True):
        """
        Module constructor.

        Args:
            n_bands (int): Number of channels of the input image.
            logits (bool): True if logits out, if False probabilities.
        """
        super().__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(
            weights="DeepLabV3_ResNet101_Weights.DEFAULT"
        )
        # 1 classe !
        self.model.classifier = DeepLabHead(2048, 1)
        self.sigmoid_layer = nn.Sigmoid()
        self.logits = logits

        if n_bands != 3:
            self.model.backbone["conv1"] = nn.Conv2d(
                n_bands,
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
        logits = self.model(x)["out"].squeeze()
        if self.logits:
            return logits
        else:
            return self.sigmoid_layer(logits)


class SegformerB0(SegformerForSemanticSegmentation):
    """
    SegformerB0 model.
    """

    def __new__(cls, n_bands="3", logits: bool = True):
        return SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=2,
            id2label={0: "background", 1: "building"},
            label2id={"background": 0, "building": 1},
        )


class SegformerB1(SegformerForSemanticSegmentation):
    """
    SegformerB1 model.
    """

    def __new__(cls, n_bands="3", logits: bool = True):
        return SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=2,
            id2label={0: "background", 1: "building"},
            label2id={"background": 0, "building": 1},
        )


class SegformerB2(SegformerForSemanticSegmentation):
    """
    SegformerB2 model.
    """

    def __new__(cls, n_bands="3", logits: bool = True):
        return SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=2,
            id2label={0: "background", 1: "building"},
            label2id={"background": 0, "building": 1},
        )


class SegformerB3(SegformerForSemanticSegmentation):
    """
    SegformerB3 model.
    """

    def __new__(cls, n_bands="3", logits: bool = True):
        return SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=2,
            id2label={0: "background", 1: "building"},
            label2id={"background": 0, "building": 1},
        )


class SegformerB4(SegformerForSemanticSegmentation):
    """
    SegformerB4 model.
    """

    def __new__(cls, n_bands="3", logits: bool = True):
        return SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=2,
            id2label={0: "background", 1: "building"},
            label2id={"background": 0, "building": 1},
        )


class SegformerB5(SegformerForSemanticSegmentation):
    """
    SegformerB5 model.
    """

    def __new__(cls, n_bands="3", logits: bool = True):
        return SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=2,
            id2label={0: "background", 1: "building"},
            label2id={"background": 0, "building": 1},
        )

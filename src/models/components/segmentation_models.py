from typing import Optional

import requests
import torch
import torchvision
from torch import nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from transformers import (
    SegformerDecodeHead,
    SegformerModel,
    SegformerPreTrainedModel,
)


class DeepLabv3Module(nn.Module):
    """
    From the paper https://arxiv.org/abs/1706.05587
    segmentation model using atrous convolution to
    take into account multiscales effect.
    """

    def __init__(self, n_bands: int = 3, logits: bool = True, freeze_encoder: bool = False):
        """
        Module constructor.

        Args:
            n_bands (int): Number of channels of the input image.
            logits (bool): True if logits out, if False probabilities.
            freeze_encoder (bool): True to freeze encoder parameters.
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

        if freeze_encoder:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward method.
        """
        logits = self.model(x)["out"]
        if self.logits:
            return logits
        else:
            return self.softmax_layer(logits)

    def freeze(self):
        """
        Freeze encoder parameters.
        """
        for param in self.model.backbone.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Unfreeze encoder parameters.
        """
        for param in self.model.backbone.parameters():
            param.requires_grad = True


class SingleClassDeepLabv3Module(nn.Module):
    """
    From the paper https://arxiv.org/abs/1706.05587
    segmentation model using atrous convolution to
    take into account multiscales effect.
    """

    def __init__(self, n_bands: int = 3, logits: bool = True, freeze_encoder: bool = False):
        """
        Module constructor.

        Args:
            n_bands (int): Number of channels of the input image.
            logits (bool): True if logits out, if False probabilities.
            freeze_encoder (bool): True to freeze encoder parameters.
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

        if freeze_encoder:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward method.
        """
        logits = self.model(x)["out"].squeeze()
        if self.logits:
            return logits
        else:
            return self.sigmoid_layer(logits)

    def freeze(self):
        """
        Freeze encoder parameters.
        """
        for param in self.model.backbone.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Unfreeze encoder parameters.
        """
        for param in self.model.backbone.parameters():
            param.requires_grad = True


class SemanticSegmentationSegformer(SegformerPreTrainedModel):
    def __init__(self, config, logits: bool = True):
        super().__init__(config)
        self.segformer = SegformerModel(config)
        self.decode_head = SegformerDecodeHead(config)
        self.logits = logits

        # Initialize weights and apply final processing
        self.post_init()

    def freeze(self):
        """
        Freeze encoder parameters.
        """
        for param in self.segformer.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Unfreeze encoder parameters.
        """
        for param in self.segformer.parameters():
            param.requires_grad = True

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Forward method.
        """
        outputs = self.segformer(
            pixel_values,
            output_attentions=False,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=True,
        )
        encoder_hidden_states = outputs.hidden_states
        logits = self.decode_head(encoder_hidden_states)

        if labels is not None:
            # upsample logits to the images' original size
            return nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            return logits


class SegformerB0(SemanticSegmentationSegformer):
    """
    SegformerB0 model.
    """

    def __new__(
        cls,
        n_bands="3",
        logits: bool = True,
        freeze_encoder: bool = False,
        type_labeler: str = "BDTOPO",
    ):
        id2label = requests.get(
            f"https://minio.lab.sspcloud.fr/projet-slums-detection/data-label/{type_labeler}/{type_labeler.lower()}-id2label.json"
        ).json()
        id2label = {int(k): v for k, v in id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
        model = SemanticSegmentationSegformer.from_pretrained(
            "nvidia/mit-b0",
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        )
        if freeze_encoder:
            model.freeze()
        return model


class SegformerB1(SemanticSegmentationSegformer):
    """
    SegformerB1 model.
    """

    def __new__(
        cls,
        n_bands="3",
        logits: bool = True,
        freeze_encoder: bool = False,
        type_labeler: str = "BDTOPO",
    ):
        id2label = requests.get(
            f"https://minio.lab.sspcloud.fr/projet-slums-detection/data-label/{type_labeler}/{type_labeler.lower()}-id2label.json"
        ).json()
        id2label = {int(k): v for k, v in id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
        model = SemanticSegmentationSegformer.from_pretrained(
            "nvidia/mit-b1",
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        )
        if freeze_encoder:
            model.freeze()
        return model


class SegformerB2(SemanticSegmentationSegformer):
    """
    SegformerB2 model.
    """

    def __new__(
        cls,
        n_bands="3",
        logits: bool = True,
        freeze_encoder: bool = False,
        type_labeler: str = "BDTOPO",
    ):
        id2label = requests.get(
            f"https://minio.lab.sspcloud.fr/projet-slums-detection/data-label/{type_labeler}/{type_labeler.lower()}-id2label.json"
        ).json()
        id2label = {int(k): v for k, v in id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
        model = SemanticSegmentationSegformer.from_pretrained(
            "nvidia/mit-b2",
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        )
        if freeze_encoder:
            model.freeze()
        return model


class SegformerB3(SemanticSegmentationSegformer):
    """
    SegformerB3 model.
    """

    def __new__(
        cls,
        n_bands="3",
        logits: bool = True,
        freeze_encoder: bool = False,
        type_labeler: str = "BDTOPO",
    ):
        id2label = requests.get(
            f"https://minio.lab.sspcloud.fr/projet-slums-detection/data-label/{type_labeler}/{type_labeler.lower()}-id2label.json"
        ).json()
        id2label = {int(k): v for k, v in id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
        model = SemanticSegmentationSegformer.from_pretrained(
            "nvidia/mit-b3",
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        )
        if freeze_encoder:
            model.freeze()
        return model


class SegformerB4(SemanticSegmentationSegformer):
    """
    SegformerB4 model.
    """

    def __new__(
        cls,
        n_bands="3",
        logits: bool = True,
        freeze_encoder: bool = False,
        type_labeler: str = "BDTOPO",
    ):
        id2label = requests.get(
            f"https://minio.lab.sspcloud.fr/projet-slums-detection/data-label/{type_labeler}/{type_labeler.lower()}-id2label.json"
        ).json()
        id2label = {int(k): v for k, v in id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
        model = SemanticSegmentationSegformer.from_pretrained(
            "nvidia/mit-b4",
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        )
        if freeze_encoder:
            model.freeze()
        return model


class SegformerB5(SemanticSegmentationSegformer):
    """
    SegformerB5 model.
    """

    def __new__(
        cls,
        n_bands="3",
        logits: bool = True,
        freeze_encoder: bool = False,
        type_labeler: str = "BDTOPO",
    ):
        id2label = requests.get(
            f"https://minio.lab.sspcloud.fr/projet-slums-detection/data-label/{type_labeler}/{type_labeler.lower()}-id2label.json"
        ).json()
        id2label = {int(k): v for k, v in id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
        model = SemanticSegmentationSegformer.from_pretrained(
            "nvidia/mit-b5",
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        )
        if freeze_encoder:
            model.freeze()
        return model

from typing import Optional
import torchvision
import torch
from torch import nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from transformers import (
    SegformerPreTrainedModel,
    SegformerModel,
    SegformerDecodeHead,
)
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip_features): 
        x = self.upconv(x)
        diffY = skip_features.size()[2] - x.size()[2]
        diffX = skip_features.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat((x, skip_features), dim=1)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_bands=3, logits=True, freeze_encoder=False):
        super(UNet, self).__init__()
        self.encoder1 = EncoderBlock(n_bands, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        self.bottleneck = ConvBlock(512, 1024)

        self.decoder1 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)

        self.conv = nn.Conv2d(64, 2, kernel_size=1)
        self.softmax_layer = nn.Softmax(dim=1)
        self.logits = logits

    def forward(self, x):
        s1, p1 = self.encoder1(x)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        s4, p4 = self.encoder4(p3)

        b = self.bottleneck(p4)

        d1 = self.decoder1(b, s4)
        d2 = self.decoder2(d1, s3)
        d3 = self.decoder3(d2, s2)
        d4 = self.decoder4(d3, s1)

        logits = self.conv(d4)

        if self.logits:
            return logits
        else:
            return self.softmax_layer(logits)

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

    def __new__(cls, n_bands="3", logits: bool = True, freeze_encoder: bool = False):
        model = SemanticSegmentationSegformer.from_pretrained(
            "nvidia/mit-b0",
            num_labels=2,
            id2label={0: "background", 1: "building"},
            label2id={"background": 0, "building": 1},
        )
        if freeze_encoder:
            model.freeze()
        return model


class SegformerB1(SemanticSegmentationSegformer):
    """
    SegformerB1 model.
    """

    def __new__(cls, n_bands="3", logits: bool = True, freeze_encoder: bool = False):
        model = SemanticSegmentationSegformer.from_pretrained(
            "nvidia/mit-b1",
            num_labels=2,
            id2label={0: "background", 1: "building"},
            label2id={"background": 0, "building": 1},
        )
        if freeze_encoder:
            model.freeze()
        return model


class SegformerB2(SemanticSegmentationSegformer):
    """
    SegformerB2 model.
    """

    def __new__(cls, n_bands="3", logits: bool = True, freeze_encoder: bool = False):
        model = SemanticSegmentationSegformer.from_pretrained(
            "nvidia/mit-b2",
            num_labels=2,
            id2label={0: "background", 1: "building"},
            label2id={"background": 0, "building": 1},
        )
        if freeze_encoder:
            model.freeze()
        return model


class SegformerB3(SemanticSegmentationSegformer):
    """
    SegformerB3 model.
    """

    def __new__(cls, n_bands="3", logits: bool = True, freeze_encoder: bool = False):
        model = SemanticSegmentationSegformer.from_pretrained(
            "nvidia/mit-b3",
            num_labels=2,
            id2label={0: "background", 1: "building"},
            label2id={"background": 0, "building": 1},
        )
        if freeze_encoder:
            model.freeze()
        return model


class SegformerB4(SemanticSegmentationSegformer):
    """
    SegformerB4 model.
    """

    def __new__(cls, n_bands="3", logits: bool = True, freeze_encoder: bool = False):
        model = SemanticSegmentationSegformer.from_pretrained(
            "nvidia/mit-b4",
            num_labels=2,
            id2label={0: "background", 1: "building"},
            label2id={"background": 0, "building": 1},
        )
        if freeze_encoder:
            model.freeze()
        return model


class SegformerB5(SemanticSegmentationSegformer):
    """
    SegformerB5 model.
    """

    def __new__(cls, n_bands="3", logits: bool = True, freeze_encoder: bool = False):
        model = SemanticSegmentationSegformer.from_pretrained(
            "nvidia/mit-b5",
            num_labels=2,
            id2label={0: "background", 1: "building"},
            label2id={"background": 0, "building": 1},
        )
        if freeze_encoder:
            model.freeze()
        return model

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
import torchvision.models as models
from torchvision.models.segmentation import deeplabv3_resnet50


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


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention = AttentionBlock(out_channels)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.attention(x)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.attention = AttentionBlock(out_channels)

    def forward(self, x, skip_features):
        x = self.upconv(x)
        diffY = skip_features.size()[2] - x.size()[2]
        diffX = skip_features.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat((x, skip_features), dim=1)
        x = self.conv_block(x)
        x = self.attention(x)
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


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SemanticSegmentationSegformerV2(SegformerPreTrainedModel):
    def __init__(self, config, logits: bool = True):
        super().__init__(config)
        self.segformer = SegformerModel(config)
        self.decode_head = SegformerDecodeHead(config)
        self.logits = logits

        # Adding SE blocks
        self.se_blocks = nn.ModuleList([SEBlock(channel) for channel in config.hidden_sizes])

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

        # Apply SE blocks to the hidden states
        enhanced_hidden_states = [se_block(hidden_state) for se_block, hidden_state in zip(self.se_blocks, encoder_hidden_states)]
        
        logits = self.decode_head(enhanced_hidden_states)

        # Upsample logits to the original image size
        logits = nn.functional.interpolate(
            logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False
        )

        if labels is not None:
            return logits
        else:
            return logits

class SegformerB5V2(SemanticSegmentationSegformerV2):
    """
    SegformerB5V2 model.
    """

    def __new__(cls, n_bands="3", logits: bool = True, freeze_encoder: bool = False):
        model = SemanticSegmentationSegformerV2.from_pretrained(
            "nvidia/mit-b5",
            num_labels=2,
            id2label={0: "background", 1: "building"},
            label2id={"background": 0, "building": 1},
        )
        if freeze_encoder:
            model.freeze()
        return model



class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        size = x.shape[2:]
        y1 = F.relu(self.bn1(self.conv1(x)))
        y2 = F.relu(self.bn2(self.conv2(x)))
        y3 = F.relu(self.bn3(self.conv3(x)))
        y4 = F.relu(self.bn4(self.conv4(x)))
        y5 = self.global_avg_pool(x)
        y5 = F.relu(self.bn5(self.conv5(y5)))
        y5 = F.interpolate(y5, size=size, mode='bilinear', align_corners=True)
        y = torch.cat([y1, y2, y3, y4, y5], dim=1)
        y = F.relu(self.bn_out(self.conv_out(y)))
        return y


class DeepLabV3(nn.Module):
    def __init__(self, n_bands=3, logits=True, freeze_encoder=False):
        super(DeepLabV3, self).__init__()
        self.num_classes = 2
        self.backbone = models.resnet50(pretrained=False, replace_stride_with_dilation=[False, True, True])
        self.aspp = ASPP(2048, 256)
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        size = x.shape[2:]
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.aspp(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return x


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(PyramidPoolingModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(in_channels, in_channels // 4, size) for size in pool_sizes])
        self.bottleneck = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def _make_stage(self, in_channels, out_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramids = [x]
        for stage in self.stages:
            out = F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=False)
            pyramids.append(out)
            print(f"Stage output shape: {out.shape}")  # Debugging line
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        print(f"Concatenated shape: {torch.cat(pyramids, dim=1).shape}")  # Debugging line
        return self.relu(output)


class PSPNet(nn.Module):
    def __init__(self, n_bands=3, logits=True, freeze_encoder=False):
        super(PSPNet, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.feat_channels = 2048

        self.layer0 = nn.Sequential(self.backbone.conv1,
                                    self.backbone.bn1,
                                    self.backbone.relu,
                                    self.backbone.maxpool)
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4
        self.logits = logits 

        self.ppm = PyramidPoolingModule(self.feat_channels, [1, 2, 3, 6])

        self.final = nn.Sequential(
            nn.Conv2d(self.feat_channels, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, 2, kernel_size=1)
        )

    def forward(self, x):
        x_size = x.size()
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.ppm(x)
        x = self.final(x)
        return F.interpolate(x, x_size[2:], mode='bilinear', align_corners=False)


class AttentionOCR(nn.Module):
    def __init__(self, in_channels):
        super(AttentionOCR, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.conv1(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.conv2(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.conv3(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out


class OCRModule(nn.Module):
    def __init__(self, num_classes, in_channels=2048):
        super(OCRModule, self).__init__()
        self.attention = AttentionOCR(in_channels)
        self.conv = nn.Conv2d(in_channels, num_classes, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.attention(x)
        x = self.conv(x)
        x = self.softmax(x)
        return x


class DeepLabV3PlusOCR(nn.Module):
    def __init__(self, n_bands=3, logits=True, freeze_encoder=False):
        super(DeepLabV3PlusOCR, self).__init__()
        num_classes = 2  # Fixed number of classes
        self.logits = logits
        self.deeplab = deeplabv3_resnet50(pretrained=True, progress=True)
        
        # Modify the input layer to accommodate the number of bands
        if n_bands != 3:
            self.deeplab.backbone.conv1 = nn.Conv2d(n_bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.deeplab.backbone.parameters():
                param.requires_grad = False

        self.deeplab.classifier[4] = nn.Conv2d(256, 2048, kernel_size=1)
        self.ocr = OCRModule(num_classes=num_classes, in_channels=2048)

    def forward(self, x):
        input_size = x.size()[2:]  # Save the original input size
        x = self.deeplab.backbone(x)['out']
        x = self.deeplab.classifier(x)
        x = self.ocr(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)  # Resize output to input size
        return x

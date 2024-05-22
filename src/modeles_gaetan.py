import torch
import torch.nn as nn
import torch.nn.functional as F


# TO DO :

# 0) création d'un data loader (sans augmentation), etre en mesure de recuperer un batch (transform)
# 1) creation modele à la main type Unet ou autre avec entrée ok sortie ok 
# 2) intégration du modèle dans le train.py (dataloader de taille réduit) -> puis demander àà thomas faria argoworkfloww
# 3) Revue de littérature + application  (plein de modeles sur hugging face recuperable )


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


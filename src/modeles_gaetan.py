import torchvision
from torch import nn


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




####

model = torchvision.models.segmentation.deeplabv3_resnet101(
            weights="DeepLabV3_ResNet101_Weights.DEFAULT"
        )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Calcul du nombre de paramètres
num_params = count_parameters(model)
print(f'Le nombre de paramètres dans le modèle est : {num_params}')

batch.keys()
output  = model(batch["pixel_values"])
output.keys()

output_final = output["out"]
output_final.shape

output_final[0][:,0,0]

model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))

softmax_layer = nn.Softmax(dim=1)
logits = True

output_modif = model(batch["pixel_values"])
output_modif = output_modif["out"]
output_modif[0][:,0,0]



# TO DO :

0) création d'un data loader (sans augmentation), etre en mesure de recuperer un batch (transform)
1) creation modele à la main type Unet ou autre avec entrée ok sortie ok 
2) intégration du modèle dans le train.py (dataloader de taille réduit) -> puis demander àà thomas faria argoworkfloww
3) Revue de littérature + application  (plein de modeles sur hugging face recuperable )


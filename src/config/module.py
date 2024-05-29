from models.components.segmentation_models import (
    UNet,
    DeepLabv3Module,
    SingleClassDeepLabv3Module,
    SegformerB0,
    SegformerB1,
    SegformerB2,
    SegformerB3,
    SegformerB4,
    SegformerB5,
    DeepLabV3
)

module_dict = {
    "deeplabv3": DeepLabv3Module,
    "single_class_deeplabv3": SingleClassDeepLabv3Module,
    "segformer-b0": SegformerB0,
    "segformer-b1": SegformerB1,
    "segformer-b2": SegformerB2,
    "segformer-b3": SegformerB3,
    "segformer-b4": SegformerB4,
    "segformer-b5": SegformerB5,
    "UNetGaetan": UNet,
    "DeepLabGaetan" : DeepLabV3
    "ASPP" : ASPP
}

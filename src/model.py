"""Segmentation model factory (U-Net / U-Net++ / DeepLabV3+) via segmentation-models-pytorch.

Outputs raw LOGITS (activation=None) — sigmoid is applied inside the loss/metrics for
numerical stability (BCEWithLogits). The ResNet34 encoder is ImageNet-pretrained, giving
strong low-level edge/texture features that transfer to thin road structures; U-Net skip
connections fuse shallow (texture) and deep (context) features — the multi-scale fusion
that lets the model infer road continuity under occlusion.
"""
import segmentation_models_pytorch as smp

_ARCH = {
    "unet": smp.Unet,
    "unetpp": smp.UnetPlusPlus,
    "deeplabv3plus": smp.DeepLabV3Plus,
}


def build_model(arch="unet", encoder="resnet34", encoder_weights="imagenet",
                in_channels=3, classes=1):
    Arch = _ARCH[arch.lower()]
    return Arch(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,  # logits
    )

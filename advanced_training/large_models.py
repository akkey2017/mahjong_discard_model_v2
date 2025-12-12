"""
Larger model variants for experiments inspired by the Qiita article.
"""

from pathlib import Path
import sys

# Ensure repository root is importable when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import (  # noqa: E402
    CoAtNet,
    ResNet,
    VisionTransformer,
    DiscardModel,
)

DEFAULT_IN_CHANNELS = 380
NUM_TILE_CLASSES = 34


def create_large_coatnet(
    in_channels=DEFAULT_IN_CHANNELS, num_classes=NUM_TILE_CLASSES, dropout=0.1
):
    """CoAtNet with wider channels and deeper transformer blocks."""
    backbone = CoAtNet(
        in_channels=in_channels,
        out_channels_list=[128, 192, 256],
        num_blocks_list=[3, 3, 6],
        expansion_factor=6,
        heads=8,
        dropout=dropout,
    )
    return DiscardModel(
        backbone,
        final_channels=256,
        num_classes=num_classes,
        dropout=dropout,
    )


def create_large_resnet(
    in_channels=DEFAULT_IN_CHANNELS, num_classes=NUM_TILE_CLASSES, dropout=0.1
):
    """Deeper ResNet-style backbone with larger channel widths."""
    backbone = ResNet(
        in_channels=in_channels,
        num_blocks_list=[3, 4, 6, 3],
        channels_list=[128, 192, 256, 320],
        dropout=dropout,
    )
    return DiscardModel(
        backbone,
        final_channels=320,
        num_classes=num_classes,
        dropout=dropout,
    )


def create_large_vit(
    in_channels=DEFAULT_IN_CHANNELS, num_classes=NUM_TILE_CLASSES, dropout=0.1
):
    """Vision Transformer variant with a larger embedding and depth."""
    backbone = VisionTransformer(
        in_channels=in_channels,
        embed_dim=512,
        depth=8,
        heads=8,
        patch_size=(1, 1),
        dropout=dropout,
    )
    return DiscardModel(
        backbone,
        final_channels=512,
        num_classes=num_classes,
        dropout=dropout,
    )


MODEL_FACTORIES = {
    "coatnet_large": create_large_coatnet,
    "resnet_large": create_large_resnet,
    "vit_large": create_large_vit,
}

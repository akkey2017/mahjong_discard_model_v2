"""
Mahjong AI Model Architectures

This module contains various neural network architectures for mahjong discard prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ==================== CoAtNet Architecture ====================

class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution block."""
    
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super().__init__()
        self.stride = stride
        hidden_dim = in_channels * expansion_factor
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        
        layers = []
        if expansion_factor != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU()
            ])
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, 
                     padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class Attention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, dim, heads=8, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """Transformer encoder block with self-attention and feed-forward network."""
    
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CoAtNet(nn.Module):
    """
    CoAtNet: Marrying Convolution and Attention for All Data Sizes
    
    Combines convolutional stages for low-level feature extraction with
    transformer stages for capturing long-range dependencies.
    """
    
    def __init__(self, in_channels, out_channels_list, num_blocks_list, 
                 expansion_factor=4, heads=4, dropout=0.0):
        super().__init__()
        
        # Stem: Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_list[0], kernel_size=3, 
                     stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels_list[0]),
            nn.GELU()
        )
        
        # Stage 1 & 2: Convolutional blocks
        self.stage1 = self._make_stage(
            MBConv, num_blocks_list[0], out_channels_list[0], 
            out_channels_list[0], 1, expansion_factor
        )
        self.stage2 = self._make_stage(
            MBConv, num_blocks_list[1], out_channels_list[0], 
            out_channels_list[1], 1, expansion_factor
        )
        
        # Stage 3: Transformer blocks
        self.stage3_conv = nn.Conv2d(out_channels_list[1], out_channels_list[2], 1)
        self.stage3_transformer = nn.Sequential(
            *[TransformerBlock(dim=out_channels_list[2], heads=heads, dropout=dropout) 
              for _ in range(num_blocks_list[2])]
        )
    
    def _make_stage(self, block, num_blocks, in_ch, out_ch, stride, expansion):
        """Create a stage with multiple blocks."""
        layers = [block(in_ch, out_ch, stride, expansion)]
        layers.extend([block(out_ch, out_ch, 1, expansion) for _ in range(num_blocks - 1)])
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (B, C, H, W) - e.g., (B, 380, 4, 9)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        
        # Convert to transformer format
        x = self.stage3_conv(x)
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.stage3_transformer(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        
        return x


class DiscardModel(nn.Module):
    """
    Mahjong discard prediction model.
    
    Takes a game state tensor and predicts which tile to discard.
    """
    
    def __init__(self, backbone, final_channels, num_classes=34, dropout=0.0):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(final_channels, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.dropout(x)
        return self.fc(x)


# ==================== ResNet Architecture ====================

class ResidualBlock(nn.Module):
    """Standard residual block for ResNet."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity)
        out = F.gelu(out)
        
        return out


class ResNet(nn.Module):
    """
    ResNet architecture adapted for mahjong state prediction.
    
    Uses residual connections to enable training of deeper networks.
    """
    
    def __init__(self, in_channels, num_blocks_list, channels_list, dropout=0.0):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels_list[0], kernel_size=3, 
                     stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels_list[0]),
            nn.GELU()
        )
        
        self.layer1 = self._make_layer(channels_list[0], channels_list[0], num_blocks_list[0], 1)
        self.layer2 = self._make_layer(channels_list[0], channels_list[1], num_blocks_list[1], 1)
        self.layer3 = self._make_layer(channels_list[1], channels_list[2], num_blocks_list[2], 1)
        
        if len(num_blocks_list) > 3 and len(channels_list) > 3:
            self.layer4 = self._make_layer(channels_list[2], channels_list[3], num_blocks_list[3], 1)
            self.final_channels = channels_list[3]
        else:
            self.layer4 = None
            self.final_channels = channels_list[2]
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a layer with multiple residual blocks."""
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        layers.extend([ResidualBlock(out_channels, out_channels, 1) 
                      for _ in range(num_blocks - 1)])
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)
        return x


# ==================== Vision Transformer Architecture ====================

class PatchEmbedding(nn.Module):
    """Convert input into patch embeddings."""
    
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, 
                                    kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.projection(x)  # (B, embed_dim, H', W')
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for mahjong state processing.
    
    Processes the entire state as a sequence of patches using transformer blocks.
    """
    
    def __init__(self, in_channels, embed_dim, depth, heads, patch_size=(1, 1), dropout=0.0):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        
        # Positional embedding for 4x9 board after patching
        num_patches = (4 // patch_size[0]) * (9 // patch_size[1])
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, heads, dropout) for _ in range(depth)]
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        self.final_channels = embed_dim
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.norm(x)
        
        # Reshape back to spatial format for compatibility
        b, n, c = x.shape
        h = 4 // self.patch_embed.patch_size[0]
        w = 9 // self.patch_embed.patch_size[1]
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        
        return x


# ==================== Model Factory Functions ====================

def create_coatnet_model(in_channels=380, num_classes=34, dropout=0.0):
    """Create a CoAtNet-based discard model with default parameters."""
    coatnet_params = {
        "in_channels": in_channels,
        "out_channels_list": [64, 96, 128],
        "num_blocks_list": [2, 2, 4],
        "expansion_factor": 4,
        "heads": 4,
        "dropout": dropout
    }
    backbone = CoAtNet(**coatnet_params)
    return DiscardModel(backbone, final_channels=128, num_classes=num_classes, dropout=dropout)


def create_resnet_model(in_channels=380, num_classes=34, dropout=0.0):
    """Create a ResNet-based discard model."""
    resnet_params = {
        "in_channels": in_channels,
        "num_blocks_list": [2, 2, 4, 2],
        "channels_list": [64, 96, 128, 160],
        "dropout": dropout
    }
    backbone = ResNet(**resnet_params)
    return DiscardModel(backbone, final_channels=160, num_classes=num_classes, dropout=dropout)


def create_vit_model(in_channels=380, num_classes=34, dropout=0.0):
    """Create a Vision Transformer-based discard model."""
    vit_params = {
        "in_channels": in_channels,
        "embed_dim": 256,
        "depth": 6,
        "heads": 8,
        "patch_size": (1, 1),
        "dropout": dropout
    }
    backbone = VisionTransformer(**vit_params)
    return DiscardModel(backbone, final_channels=256, num_classes=num_classes, dropout=dropout)

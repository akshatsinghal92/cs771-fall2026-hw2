import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .blocks import (PatchEmbed, window_partition, window_unpartition,
                    DropPath, MLP, trunc_normal_)


################################################################################
# You will need to fill in the missing code in this file
################################################################################


################################################################################
# Part I.1: A simple convolutional neural network
################################################################################
class SimpleNet(nn.Module):
    # a simple CNN for image classifcation
    def __init__(self, conv_op=nn.Conv2d, num_classes=100):
        super(SimpleNet, self).__init__()
        # you can start from here and create a better model
        self.features = nn.Sequential(
            # conv1 block: conv 7x7
            conv_op(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv2 block: simple bottleneck
            conv_op(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv_op(64, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv3 block: simple bottleneck
            conv_op(256, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            conv_op(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv_op(128, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        # global avg pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def reset_parameters(self):
        # init all params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
################################################################################
# Part I.2: Design your own CNN
################################################################################
# define your own network here

# change this to your model!
default_cnn_model = SimpleNet

################################################################################
# Part I.3: Understanding self-attention and Transformer block
################################################################################
class Attention(nn.Module):
    """Multi-head Self-Attention."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
    ):
        """
        Args:
            dim (int): Number of input channels. We assume Q, K, V will be of
                same dimension as the input.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # linear projection for query, key, value
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # linear projection at the end
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # input size (B, H, W, C)
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (
            self.qkv(x).reshape(
                B, H * W, 3, self.num_heads, -1
            ).permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        #################################################################################
        # Fill in the code here
        #################################################################################
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (
            (attn @ v)
            .view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer blocks with support of local window self-attention"""
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        window_size=0,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            window_size (int): Window size for window attention blocks.
                If it equals 0, then global attention is used.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.drop_path = DropPath(drop_path) if drop_path>0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer
        )

        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)

        ########################################################################
        # Fill in the code here
        ########################################################################
        # The implementation shall support local self-attention
        # (also known as window attention)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)

        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


#################################################################################
# Part I.4: Design and train a vision Transformer
#################################################################################
class SimpleViT(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size=128,
        num_classes=100,
        patch_size=16,
        in_chans=3,
        embed_dim=192,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        window_size=4,
        window_block_indexes=(0, 2),
    ):
        """
        Args:
            img_size (int): Input image size.
            num_classes (int): Number of object categories
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            window_size (int): Window size for local attention blocks.
            window_block_indexes (list): Indexes for blocks using local attention.
                Local window attention allows more efficient computation, and
                can be coupled with standard global attention. E.g., [0, 2]
                indicates the first and the third blocks will use local window
                attention, while other block use standard attention.

        Feel free to modify the default parameters here.
        """
        super(SimpleViT, self).__init__()
        assert (num_heads > 0) and (embed_dim % num_heads == 0) 

        if use_abs_pos:
            # Initialize absolute positional embedding with image size
            # The embedding is learned from data
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, img_size // patch_size, img_size // patch_size, embed_dim
                )
            )
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # patch embedding layer
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        ########################################################################
        # Fill in the code here
        ########################################################################
        # The implementation shall define some Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                window_size=window_size if i in window_block_indexes else 0,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, num_classes)
            if num_classes and num_classes > 0
            else nn.Identity()
        )
        self.fc = self.head


        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)
        # add any necessary weight initialization here

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        ########################################################################
        # Fill in the code here
        ########################################################################
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            if self.pos_embed.shape[1:3] == x.shape[1:3]:
                x = x + self.pos_embed
            else:
                pos_embed = self.pos_embed.permute(0, 3, 1, 2)
                pos_embed = F.interpolate(
                    pos_embed, size=x.shape[1:3], mode="bicubic", align_corners=False
                )
                pos_embed = pos_embed.permute(0, 2, 3, 1)
                x = x + pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = x.mean(dim=(1, 2))
        x = self.head(x)
        return x

# change this to your model!
default_vit_model = SimpleViT




"""
KalaKriti — Real-Time Multi-Style Neural Style Transfer
Model: TransformerNet with Conditional Instance Normalization (CIN).

Architecture:
  - Input: (B, 3, H, W) normalized to [0, 255]
  - 3 Reflection-padded Conv Blocks (downsampling by factor of 4)
  - 5 Residual Blocks with 128 channels and CIN
  - 2 Nearest-Neighbor Upsampling Blocks (upsampling by factor of 4, no checkerboard artifacts)
  - Output projection with Sigmoid scaling to [0, 255]
"""

import torch
import torch.nn as nn


class ConditionalInstanceNorm2d(nn.Module):
    """
    Conditional Instance Normalization (CIN) layer.
    Standardizes feature activations per channel across spatial dimensions (H, W),
    then modulates activations using learned style-specific scale (gamma) and shift (beta) vectors.
    """

    def __init__(self, num_features: int, num_styles: int = 5, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.num_styles = num_styles
        self.eps = eps

        # Embedding tables mapping style index to affine parameters (gamma, beta)
        self.gamma = nn.Embedding(num_styles, num_features)
        self.beta = nn.Embedding(num_styles, num_features)

        # Initialize gamma to 1.0 (identity scaling) and beta to 0.0 (identity shift)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, H, W)
            style_id: LongTensor of shape (B,) or scalar style index in [0, num_styles - 1]
        Returns:
            Normalized and conditioned Tensor of shape (B, C, H, W)
        """
        if style_id.dim() == 0:
            style_id = style_id.repeat(x.size(0))
        elif style_id.size(0) == 1 and x.size(0) > 1:
            style_id = style_id.repeat(x.size(0))

        # Spatial normalization per channel
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        normalized = (x - mean) / std

        # Retrieve affine modulation parameters: (B, C, 1, 1)
        gamma = self.gamma(style_id).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(style_id).unsqueeze(-1).unsqueeze(-1)

        return gamma * normalized + beta


class ConvBlock(nn.Module):
    """
    Reflection-padded Convolution + Conditional Instance Normalization + optional ReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        num_styles: int = 5,
        relu: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
        self.cin = ConditionalInstanceNorm2d(out_channels, num_styles=num_styles)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        out = self.pad(x)
        out = self.conv(out)
        out = self.cin(out, style_id)
        if self.relu is not None:
            out = self.relu(out)
        return out


class ResidualBlock(nn.Module):
    """
    Residual Block with two 3x3 Conv layers, CIN, and an identity skip connection.
    Preserves structural features and semantic anatomy across deep transformations.
    """

    def __init__(self, channels: int = 128, num_styles: int = 5):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=3, stride=1, num_styles=num_styles, relu=True)
        self.conv2 = ConvBlock(channels, channels, kernel_size=3, stride=1, num_styles=num_styles, relu=False)

    def forward(self, x: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x, style_id), style_id)


class UpsampleConvBlock(nn.Module):
    """
    Nearest-neighbor 2x upsampling followed by reflection-padded convolution and CIN.
    Completely eliminates checkerboard artifacts that occur with ConvTranspose2d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_styles: int = 5,
        scale_factor: int = 2,
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=kernel_size, stride=1, num_styles=num_styles, relu=True)

    def forward(self, x: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        out = self.upsample(x)
        out = self.conv(out, style_id)
        return out


class TransformerNet(nn.Module):
    """
    KalaKriti Feed-Forward Generator Network.
    Transforms an input image of arbitrary resolution into the conditioned style.
    """

    def __init__(self, num_styles: int = 5, base_channels: int = 32):
        super().__init__()
        self.num_styles = num_styles
        self.base_channels = base_channels

        # Initial Downsampling layers
        self.conv1 = ConvBlock(3, base_channels, kernel_size=9, stride=1, num_styles=num_styles, relu=True)
        self.conv2 = ConvBlock(base_channels, base_channels * 2, kernel_size=3, stride=2, num_styles=num_styles, relu=True)
        self.conv3 = ConvBlock(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, num_styles=num_styles, relu=True)

        # 5 Residual Blocks (128 channels when base_channels=32)
        res_channels = base_channels * 4
        self.res1 = ResidualBlock(res_channels, num_styles=num_styles)
        self.res2 = ResidualBlock(res_channels, num_styles=num_styles)
        self.res3 = ResidualBlock(res_channels, num_styles=num_styles)
        self.res4 = ResidualBlock(res_channels, num_styles=num_styles)
        self.res5 = ResidualBlock(res_channels, num_styles=num_styles)

        # Upsampling layers
        self.up1 = UpsampleConvBlock(res_channels, base_channels * 2, kernel_size=3, num_styles=num_styles)
        self.up2 = UpsampleConvBlock(base_channels * 2, base_channels, kernel_size=3, num_styles=num_styles)

        # Final output projection
        self.final_pad = nn.ReflectionPad2d(4)
        self.final_conv = nn.Conv2d(base_channels, 3, kernel_size=9, stride=1)

    def forward(self, x: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor of shape (B, 3, H, W) in range [0, 255]
            style_id: LongTensor of shape (B,) or scalar integer style index
        Returns:
            Output image tensor of shape (B, 3, H, W) strictly bounded in range [0, 255]
        """
        if not isinstance(style_id, torch.Tensor):
            style_id = torch.tensor(style_id, dtype=torch.long, device=x.device)
        elif style_id.device != x.device:
            style_id = style_id.to(x.device)

        # Downsampling
        h = self.conv1(x, style_id)
        h = self.conv2(h, style_id)
        h = self.conv3(h, style_id)

        # Residual transformations
        h = self.res1(h, style_id)
        h = self.res2(h, style_id)
        h = self.res3(h, style_id)
        h = self.res4(h, style_id)
        h = self.res5(h, style_id)

        # Upsampling
        h = self.up1(h, style_id)
        h = self.up2(h, style_id)

        # Final projection to RGB
        h = self.final_pad(h)
        h = self.final_conv(h)

        # Sigmoid projection strictly guarantees valid RGB range [0, 255] without clipping artifacts
        return torch.sigmoid(h) * 255.0

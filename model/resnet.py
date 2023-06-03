import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple[int, int] | int = 3,
                 stride: int = 1,
                 padding: int | str = "same") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample: None | nn.Module = None
        if stride != 0:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride
                ),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out+residual)


class ResNet(nn.Module):
    def __init__(self, first_conv: dict, block_list: list[dict], pool_size: int) -> None:
        super().__init__()
        self.first_conv = nn.Conv2d(**first_conv)
        self.res_blocks = nn.ModuleList([
            ResBlock(**block_conf) for block_conf in block_list
        ])
        self.pooling = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.first_conv(x)
        for block in self.res_blocks:
            out = block(out)
        out = self.pooling(out)
        return torch.flatten(out, start_dim=1)

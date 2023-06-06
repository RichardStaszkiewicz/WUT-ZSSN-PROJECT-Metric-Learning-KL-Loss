from typing import Callable
import torch
import torch.nn as nn


class SLPBlock(nn.Module):
    """
    _activation_fun_dict -> dict[str, callable]
    """
    _activation_fun_dict: dict = {
        "relu": nn.functional.relu,
        "tanh": nn.functional.tanh,
        "sigmoid": nn.functional.sigmoid,
        "none": lambda x: x,
    }

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 activation_fun: str = "relu",
                 batch_norm: bool = True,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.layer = nn.Linear(in_size, out_size)
        self.bn = nn.BatchNorm1d(out_size) if batch_norm is True else None
        assert activation_fun in self._activation_fun_dict.keys()
        self.act_fun = self._activation_fun_dict[activation_fun]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer(x)
        if self.bn is not None:
            out = self.bn(out)
        out = self.act_fun(out)
        return self.dropout(out)


class MLP(nn.Module):
    def __init__(self, block_list: list) -> None:
        """
        block_list -> list[dict]
        """
        super().__init__()
        self.blocks = nn.ModuleList([
            SLPBlock(**block_conf) for block_conf in block_list
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            out = block(out)
        return out

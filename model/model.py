from typing import Any, Callable
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import pytorch_lightning as pl
import numpy as np
from .resnet import ResNet
from .mlp import MLP
from pytorch_metric_learning import distances


def KL_d(emb1: tuple,
         emb2: tuple) -> torch.Tensor:
    """
    emb1 -> tuple[torch.Tensor, torch.Tensor]
    emb2 -> tuple[torch.Tensor, torch.Tensor]
    """
    m1, s1 = emb1
    m2, s2 = emb2
    return (
        ((2 * s1).exp() + (m1 - m2).pow(2)) * 0.5 * (-2 * s2).exp() +
        ((2 * s2).exp() + (m1 - m2).pow(2)) * 0.5 * (-2 * s1).exp() - 1
    ).sum()


def KL_dreg(emb1: tuple,
            emb2: tuple) -> torch.Tensor:
    """
    emb1 -> tuple[torch.Tensor, torch.Tensor]
    emb2 -> tuple[torch.Tensor, torch.Tensor]
    """
    m1, s1 = emb1
    m2, s2 = emb2
    return (
        ((2 * s1).exp() + m1.pow(2)) * 0.5 +
        (1 + m1.pow(2)) * 0.5 * (-2 * s1).exp() - 1 +
        ((2 * s2).exp() + m2.pow(2)) * 0.5 +
        (1 + m2.pow(2)) * 0.5 * (-2 * s2).exp() - 1
    ).sum()


def loss_same_class(emb1: tuple,
                    emb2: tuple,
                    alpha: float,
                    m: float) -> torch.Tensor:
    """
    emb1 -> tuple[torch.Tensor, torch.Tensor]
    emb2 -> tuple[torch.Tensor, torch.Tensor]
    """
    return (KL_d(emb1, emb2) + alpha * KL_dreg(emb1, emb2))


def loss_different_class(emb1: tuple,
                         emb2: tuple,
                         alpha: float,
                         m: float) -> torch.Tensor:
    """
    emb1 -> tuple[torch.Tensor, torch.Tensor]
    emb2 -> tuple[torch.Tensor, torch.Tensor]
    """
    x = alpha * KL_dreg(emb1, emb2)
    return (torch.max(m - KL_d(emb1, emb2), torch.zeros_like(x)) + x)


def random_class_pairs(embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # strategy - iterate through batch. Match two consecutive results.
    # embeds size = [batch size, no. output dims, 2 (for each dim mean and std)]
    # labels = [batch size]
    loss = torch.tensor(0, dtype=float, device=embeds[0][0].device)
    for i in range(len(embeds)):
        j = (i + 1) % len(embeds)
        if labels[i] == labels[j]:
            loss += loss_same_class((embeds[i].T[0], embeds[i].T[1]), (embeds[j].T[0], embeds[j].T[1]), 0.5, 0.5)
        else:
            loss += loss_different_class((embeds[i].T[0], embeds[i].T[1]), (embeds[j].T[0], embeds[j].T[1]), 0.5, 0.5)
    return loss

class KLDistance(distances.BaseDistance):
        def __init__(self, normalize_embeddings=True, p=2, power=1, is_inverted=False, **kwargs):
            super().__init__(normalize_embeddings, p, power, is_inverted, **kwargs)

        def compute_mat(self, query_emb, ref_emb):
            # Must return a matrix where mat[j,k] represents
            # the distance/similarity between query_emb[j] and ref_emb[k]
            ans = []
            for j in range(len(query_emb)):
                a = []
                for k in range(len(ref_emb)):
                    a.append(self._dist(query_emb[j], ref_emb[k]))
                ans.append(a)
            return torch.Tensor(np.array(ans))

        def pairwise_distance(self, query_emb, ref_emb):
            # Must return a tensor where output[j] represents
            # the distance/similarity between query_emb[j] and ref_emb[j]
            ans = []
            for i in range(len(query_emb)):
                ans.append(self._dist(query_emb[i], ref_emb[i]))
            return torch.Tensor(np.array(ans))

        def _dist(self, emb1, emb2):
            m1, s1 = (emb1[:len(emb1)//2], emb1[len(emb1)//2:])
            m2, s2 = (emb2[:len(emb2)//2], emb2[len(emb2)//2:])
            return (
                ((2 * s1).exp() + (m1 - m2).pow(2)) * 0.5 * (-2 * s2).exp() +
                ((2 * s2).exp() + (m1 - m2).pow(2)) * 0.5 * (-2 * s1).exp() - 1
            ).sum()


class KLLossMetricLearning(pl.LightningModule):
    """
    _batch_handlers -> dict[str, callable]
    """
    _batch_handlers: dict = {
        "random_class_pairs": random_class_pairs
    }

    def __init__(self,
                 exp_class_distance: float,
                 regularization_ratio: float,
                 batch_handling: str,
                 resnet_config: dict,
                 mlp_config: dict,
                 lr: float,
                 img_key: Any = 0,
                 class_key: Any = 1,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.resnet = ResNet(**resnet_config)
        self.mlp = MLP(**mlp_config)
        self.exp_class_distance = exp_class_distance
        self.regularization_ratio = regularization_ratio
        assert batch_handling in self._batch_handlers.keys()
        self.batch_handler = self._batch_handlers[batch_handling]
        self.img_key = img_key
        self.class_key = class_key

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        imgs = batch[self.img_key]
        labels = batch[self.class_key]
        embeds = self(imgs)
        loss = self.batch_handler(embeds, labels)

        loss_dict = {"train/loss": loss}
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        imgs = batch[self.img_key]
        labels = batch[self.class_key]
        embeds = self(imgs)
        loss = self.batch_handler(embeds, labels)

        loss_dict = {"val/loss": loss}
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return loss

    def forward(self, imgs: torch.Tensor) -> tuple:
        """
        return -> tuple[torch.Tensor, torch.Tensor]
        """
        out = self.resnet(imgs)
        out = self.mlp(out)
        latent_dim = out.shape[1] // 2
        mean, std = out[:, :latent_dim], out[:, latent_dim:]
        return mean, std

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        imgs = batch[self.img_key]
        labels = batch[self.class_key]
        embeds = self(imgs)
        loss = self.batch_handler(embeds, labels)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optim

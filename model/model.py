from typing import Any, Callable
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import pytorch_lightning as pl
import numpy as np
from .resnet import ResNet
from .mlp import MLP
from pytorch_metric_learning import distances, miners, losses


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


def random_class_pairs(embeds: torch.Tensor, labels: torch.Tensor, exp_class_distance: float=None, regularization_ratio: float=None) -> torch.Tensor:
    # strategy - iterate through batch. Match two consecutive results.
    # embeds size = [batch size, no. output dims, 2 (for each dim mean and std)]
    # labels = [batch size]
    # exp_class_distance = default 1
    # regularization_ratio = default 0.2
    exp_class_distance = exp_class_distance if exp_class_distance else 1
    regularization_ratio = regularization_ratio if regularization_ratio else 0.2

    loss = torch.tensor(0, dtype=float, device=embeds[0][0].device)
    for i in range(len(embeds)):
        j = (i + 1) % len(embeds)
        if labels[i] == labels[j]:
            loss += loss_same_class(
                emb1 = (embeds[i].T[0], embeds[i].T[1]),
                emb2 = (embeds[j].T[0], embeds[j].T[1]),
                alpha=regularization_ratio, m=exp_class_distance)
        else:
            loss += loss_different_class(
                emb1 = (embeds[i].T[0], embeds[i].T[1]),
                emb2 = (embeds[j].T[0], embeds[j].T[1]),
                alpha=regularization_ratio, m=exp_class_distance)
    return loss


def pair_margin_miner(embeds: torch.Tensor, labels: torch.Tensor, exp_class_distance: float=None, regularization_ratio: float=None) -> torch.Tensor:
    # strategy - get only pairs which are not distanced enough
    # embeds size = [batch size, 2 * output dims (means+std)]
    # labels = [batch size]
    # exp_class_distance = default 1
    # regularization_ratio = default 0.2
    miner_func = miners.PairMarginMiner(pos_margin=0.2, neg_margin=0.8, distance=KLDistance())
    loss_func = KLoss(distance=KLDistance())
    miner_output = miner_func(embeds, labels)
    return loss_func(embeds,labels, miner_output)

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

class KLoss(losses.BaseMetricLossFunction):
    def __init__(self, embedding_regularizer=None, embedding_reg_weight=1, pos_negative_ratio=None, **kwargs):
        super().__init__(embedding_regularizer, embedding_reg_weight, **kwargs)
        self.pos_negative_ratio = pos_negative_ratio

    def cnt_ratios(self, pos, neg, ratio):
        if not ratio:
            return pos, neg
        positive_count = pos
        negative_count = positive_count // ratio
        while negative_count > neg:
            positive_count = positive_count - 1
            negative_count = positive_count // ratio
        return positive_count, negative_count

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        if self.pos_negative_ratio:
            positive_count, negative_count = self.cnt_ratios(len(indices_tuple[0]), len(indices_tuple[2]), self.pos_negative_ratio)
        print("positive count", positive_count, "\nnegative count", negative_count)
        print(embeddings.shape)
        print(labels.shape)
        print(indices_tuple)
        print(ref_emb)
        print(ref_labels)
        raise NotImplementedError

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
        self.batch_handling_name = self.batch_handling
        self.batch_handler = self._batch_handlers[batch_handling]
        self.img_key = img_key
        self.class_key = class_key

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        imgs = batch[self.img_key]
        labels = batch[self.class_key]
        means, stds = self(imgs)
        if self.batch_handling_name == "random_class_pairs":
            embeds = torch.cat((means.unsqueeze(2), stds.unsqueeze(2)), dim=2)
        elif self.batch_handling_name == "pair_margin_miner":
            embeds = torch.cat((means, stds), dim=1)
        loss = self.batch_handler(embeds, labels, self.exp_class_distance, self.regularization_ratio)

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

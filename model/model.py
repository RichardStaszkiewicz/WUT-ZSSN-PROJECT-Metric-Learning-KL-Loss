from typing import Any
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


def kl_knn_func(query: list[tuple[torch.Tensor, torch.Tensor]],
                k: int) -> tuple[torch.Tensor, torch.Tensor]:
    distances = None
    indices = None
    for i, point in enumerate(query):
        kl_distances = {l: KL_d(point, other_emb)
                        for l, other_emb in [(j, query[j])
                                             for j in range(len(query)) if j != i]}
        kl_distances = {key: v for key, v in sorted(kl_distances.items(), key=lambda item: item[1])}
        ind = torch.tensor(list(kl_distances.keys())[:k])
        dist = torch.tensor([kl_distances[int(i)] for i in ind])
        indices = torch.vstack([indices, ind]) if indices is not None else ind
        distances = torch.vstack([distances, dist]) if distances is not None else dist
    return distances, indices


def recall_at_k(query: list[tuple[torch.Tensor, torch.Tensor]],
                labels: torch.Tensor,
                k: list[int] | int) -> dict[int, float]:
    if isinstance(k, int):
        k = [k]
    _, indices = kl_knn_func(query, max(k))
    adj_labels = torch.hstack([labels.unsqueeze(dim=1) for _ in range(max(k))])
    recall_dict = dict()
    for i in k:
        recall_dict[i] = float(
            (labels[indices[:, :i]] == adj_labels[:, :i]).any(dim=1).sum() / len(labels))
    return recall_dict


def precision_at_k(query: list[tuple[torch.Tensor, torch.Tensor]],
                   labels: torch.Tensor,
                   k: list[int] | int) -> dict[int, float]:
    if isinstance(k, int):
        k = [k]
    _, indices = kl_knn_func(query, max(k))
    adj_labels = torch.hstack([labels.unsqueeze(dim=1) for _ in range(max(k))])
    precision_dict = dict()
    for i in k:
        precision_dict[i] = float(
            (labels[indices[:, :i]] == adj_labels[:, :i]).sum() / adj_labels[:, :i].numel())
    return precision_dict


def random_class_pairs(embeds: torch.Tensor,
                       labels: torch.Tensor,
                       exp_class_distance: float | None = None,
                       regularization_ratio: float | None = None) -> torch.Tensor:
    # strategy - iterate through batch. Match two consecutive results.
    # embeds size = [batch size, no. output dims, 2 (for each dim mean and std)]
    # labels = [batch size]
    # exp_class_distance = default 1
    # regularization_ratio = default 0.2
    exp_class_distance = exp_class_distance if exp_class_distance else 1
    regularization_ratio = regularization_ratio if regularization_ratio else 0.2

    loss = torch.tensor(0, dtype=torch.float, device=embeds[0][0].device)
    for i in range(len(embeds)):
        j = (i + 1) % len(embeds)
        if labels[i] == labels[j]:
            loss += loss_same_class(
                emb1=(embeds[i].T[0], embeds[i].T[1]),
                emb2=(embeds[j].T[0], embeds[j].T[1]),
                alpha=regularization_ratio, m=exp_class_distance)
        else:
            loss += loss_different_class(
                emb1=(embeds[i].T[0], embeds[i].T[1]),
                emb2=(embeds[j].T[0], embeds[j].T[1]),
                alpha=regularization_ratio, m=exp_class_distance)
    return loss


def pair_margin_miner(embeds: torch.Tensor,
                      labels: torch.Tensor,
                      exp_class_distance: float | None = None,
                      regularization_ratio: float | None = None) -> torch.Tensor:
    # strategy - get only pairs which are not distanced enough
    # embeds size = [batch size, 2 * output dims (means+std)]
    # labels = [batch size]
    # exp_class_distance = default 1
    # regularization_ratio = default 0.2
    exp_class_distance = exp_class_distance if exp_class_distance else 1
    regularization_ratio = regularization_ratio if regularization_ratio else 0.2

    miner_func = miners.PairMarginMiner(
        pos_margin=exp_class_distance, neg_margin=exp_class_distance, distance=KLDistance())
    loss_func = KLoss(distance=KLDistance(),
                      exp_class_distance=exp_class_distance,
                      regularization_ratio=regularization_ratio)
    miner_output = miner_func(embeds.cpu(), labels)
    miner_output = tuple([output.to(labels.device) for output in miner_output])
    return loss_func(embeds, labels, miner_output)


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
                a.append(self._dist(query_emb[j], ref_emb[k]).cpu())
            ans.append(a)
        return torch.tensor(ans)

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
    def __init__(self,
                 embedding_regularizer=None,
                 exp_class_distance=None,
                 regularization_ratio=None,
                 embedding_reg_weight=1,
                 pos_negative_ratio=None,
                 **kwargs) -> None:
        super().__init__(embedding_regularizer, embedding_reg_weight, **kwargs)
        self.pos_negative_ratio = pos_negative_ratio
        self.m = exp_class_distance
        self.alpha = regularization_ratio

    def cnt_ratios(self, pos, neg, ratio):
        if not ratio:
            return pos, neg
        negative_count = pos // ratio
        positive_count = int(negative_count * ratio)
        while negative_count > neg:
            positive_count = positive_count - 1
            negative_count = positive_count // ratio
        return positive_count, negative_count

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        positive_count, negative_count = self.cnt_ratios(
            len(indices_tuple[0]), len(indices_tuple[2]), self.pos_negative_ratio)
        loss = torch.tensor(0, dtype=float, device=embeddings[0].device)
        pivot = len(embeddings[0]) // 2
        for i in range(positive_count):
            loss += loss_same_class(
                emb1=(embeddings[indices_tuple[0][i]][:pivot],
                      embeddings[indices_tuple[0][i]][pivot:]),
                emb2=(embeddings[indices_tuple[1][i]][:pivot],
                      embeddings[indices_tuple[1][i]][pivot:]),
                alpha=self.alpha,
                m=self.m
            )
        for i in range(negative_count):
            loss += loss_different_class(
                emb1=(embeddings[indices_tuple[2][i]][:pivot],
                      embeddings[indices_tuple[2][i]][pivot:]),
                emb2=(embeddings[indices_tuple[3][i]][:pivot],
                      embeddings[indices_tuple[3][i]][pivot:]),
                alpha=self.alpha,
                m=self.m
            )
        return {
            'loss': {
                'losses': loss,
                'indices': None,
                'reduction_type': 'already_reduced'
                }
            }


class KLLossMetricLearning(pl.LightningModule):
    """
    _batch_handlers -> dict[str, callable]
    """
    _batch_handlers: dict = {
        "random_class_pairs": random_class_pairs,
        "pair_margin_miner": pair_margin_miner
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
                 precision_k: list[int] | int = 1,
                 recall_k: list[int] | int = 1,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.resnet = ResNet(**resnet_config)
        self.mlp = MLP(**mlp_config)
        self.exp_class_distance = exp_class_distance
        self.regularization_ratio = regularization_ratio
        assert batch_handling in self._batch_handlers.keys()
        self.batch_handling_name = batch_handling
        self.batch_handler = self._batch_handlers[batch_handling]
        self.img_key = img_key
        self.class_key = class_key
        self.precision_k = precision_k
        self.recall_k = recall_k

        # self.train_emb = None
        # self.train_labels = None
        # self.val_emb = None
        # self.val_labels = None

    def get_embeds(self, means, stds):
        if self.batch_handling_name == "random_class_pairs":
            return torch.cat((means.unsqueeze(2), stds.unsqueeze(2)), dim=2)
        elif self.batch_handling_name == "pair_margin_miner":
            return torch.cat((means, stds), dim=1)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        imgs = batch[self.img_key]
        labels = batch[self.class_key]
        means, stds = self(imgs)
        embeds = self.get_embeds(means, stds)
        loss = self.batch_handler(
            embeds, labels, self.exp_class_distance, self.regularization_ratio)

        # self.train_emb = torch.cat(
        #     [self.train_emb, embeds], dim=0) if self.train_emb is not None else embeds
        # self.train_labels = torch.cat(
        #     [self.train_labels, labels], dim=0) if self.train_labels is not None else labels

        if isinstance(embeds, torch.Tensor):
            hidden = embeds.shape[1] // 2
            embeds = [(emb[:hidden], emb[hidden:]) for emb in embeds]
        prec_dict = precision_at_k(embeds, labels, self.precision_k)
        log_dict = {f"train/precision@{k}": prec_dict[k] for k in prec_dict.keys()}
        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        recall_dict = recall_at_k(embeds, labels, self.recall_k)
        log_dict = {f"train/recall@{k}": recall_dict[k] for k in recall_dict.keys()}
        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        loss_dict = {"train/loss": loss}
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    # def on_train_epoch_end(self) -> None:
    #     prec_dict = precision_at_k(self.train_emb, self.train_labels, self.precision_k)
    #     log_dict = {f"train/precision@{k}": prec_dict[k] for k in prec_dict.keys()}
    #     self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    #     recall_dict = recall_at_k(self.train_emb, self.train_labels, self.recall_k)
    #     log_dict = {f"train/recall@{k}": recall_dict[k] for k in recall_dict.keys()}
    #     self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    #     self.train_emb = None
    #     self.train_labels = None

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        imgs = batch[self.img_key]
        labels = batch[self.class_key]
        means, stds = self(imgs)
        embeds = self.get_embeds(means, stds)
        loss = self.batch_handler(embeds, labels)

        # self.val_emb = torch.cat(
        #     [self.val_emb, embeds], dim=0) if self.val_emb is not None else embeds
        # self.val_labels = torch.cat(
        #     [self.val_labels, labels], dim=0) if self.val_labels is not None else labels

        if isinstance(embeds, torch.Tensor):
            hidden = embeds.shape[1] // 2
            embeds = [(emb[:hidden], emb[hidden:]) for emb in embeds]
        prec_dict = precision_at_k(embeds, labels, self.precision_k)
        log_dict = {f"val/precision@{k}": prec_dict[k] for k in prec_dict.keys()}
        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        recall_dict = recall_at_k(embeds, labels, self.recall_k)
        log_dict = {f"val/recall@{k}": recall_dict[k] for k in recall_dict.keys()}
        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        loss_dict = {"val/loss": loss}
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return loss

    # def on_validation_epoch_end(self) -> None:
    #     if isinstance(self.val_emb, torch.Tensor):
    #         hidden = self.val_emb.shape[1] // 2
    #         self.val_emb = [(emb[:hidden], emb[hidden:]) for emb in self.val_emb]
    #     prec_dict = precision_at_k(self.val_emb, self.val_labels, self.precision_k)
    #     log_dict = {f"val/precision@{k}": prec_dict[k] for k in prec_dict.keys()}
    #     self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    #     recall_dict = recall_at_k(self.val_emb, self.val_labels, self.recall_k)
    #     log_dict = {f"val/recall@{k}": recall_dict[k] for k in recall_dict.keys()}
    #     self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    #     self.val_emb = None
    #     self.val_labels = None

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

        means, stds = self(imgs)
        embeds = self.get_embeds(means, stds)
        loss = self.batch_handler(embeds, labels)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optim

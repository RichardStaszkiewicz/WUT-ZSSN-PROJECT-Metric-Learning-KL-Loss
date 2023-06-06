import os
import torch
from omegaconf import OmegaConf
from model.model import KLLossMetricLearning, loss_different_class, loss_same_class, KL_d, KL_dreg


if __name__ == "__main__":
    conf_path = os.path.join("configs", "model.yaml")
    conf = OmegaConf.load(conf_path)
    model = KLLossMetricLearning(**conf.get("model"))
    images = torch.ones(64, 3, 32, 32)
    model(images)
    # m1 = torch.tensor([[1, 1, 1]])
    # m2 = torch.tensor([[1, 1, 1]])
    # s1 = torch.tensor([[1, 1, 1]])
    # s2 = torch.tensor([[1, 1, 1]])
    means = torch.Tensor([
        [3, 0, 3],
        [2, 0, 2],
        [7, 3, 2]
        ])
    stds = torch.Tensor([
        [0.2, 0.3, 0.1],
        [0.3, 0.1, 2],
        [0.1, 0.1, 0.1]
    ])
    labels = torch.Tensor(
        [2, 2, 3]
    )
    emb = torch.cat((means.unsqueeze(2), stds.unsqueeze(2)), dim=2)
    m1, s1 = (emb[0].T[0], emb[0].T[1])
    m2, s2 = (emb[1].T[0], emb[1].T[1])
    print(m1)
    print(s1)
    print(KL_d((m1, s1), (m2, s2)))
    print(KL_dreg((m1, s1), (m2, s2)))
    print(loss_same_class((m1, s1), (m2, s2), alpha = 1, m=None))
    print(loss_different_class((m1, s1), (m2, s2), alpha = 1, m = 1))

    #emb[n] = [[mn0, sn0], [mn1, sn1]... [mnx, snx]]
    loss = 0
    for i in range(len(emb)):
        j = (i + 1) % len(emb)
        if labels[i] == labels[j]:
            loss += loss_same_class((emb[i].T[0], emb[i].T[1]), (emb[j].T[0], emb[j].T[1]), 0.5, 0.5)
        else:
            loss += loss_different_class((emb[i].T[0], emb[i].T[1]), (emb[j].T[0], emb[j].T[1]), 0.5, 0.5)
    print(loss)
    #print(random_class_pairs(torch.Tensor(means, stds), labels))

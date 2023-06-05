import os
import torch
from omegaconf import OmegaConf
from model.model import KLLossMetricLearning, loss_different_class, loss_same_class


if __name__ == "__main__":
    conf_path = os.path.join("configs", "model.yaml")
    conf = OmegaConf.load(conf_path)
    model = KLLossMetricLearning(**conf.get("model"))
    images = torch.ones(64, 3, 32, 32)
    model(images)
    m1 = torch.tensor([[1, 1, 1], [1, 1, 1]])
    m2 = torch.tensor([[1, 1, 1], [1, 1, 1]])
    s1 = torch.tensor([[1, 1, 1], [1, 1, 1]])
    s2 = torch.tensor([[1, 1, 1], [1, 1, 1]])
    print(loss_same_class((m1, s1), (m2, s2), 0.5, 1))
    print(loss_different_class((m1, s1), (m2, s2), 0.5, 1))

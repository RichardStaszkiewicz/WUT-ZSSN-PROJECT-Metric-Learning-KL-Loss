import os
import torch
from omegaconf import OmegaConf
from model.model import KLLossMetricLearning


if __name__ == "__main__":
    conf_path = os.path.join("configs", "model.yaml")
    conf = OmegaConf.load(conf_path)
    model = KLLossMetricLearning(**conf.get("model"))
    images = torch.ones(64, 3, 32, 32)
    model(images)

from os import path
import torch
# from omegaconf import OmegaConf
# from model.model import KLLossMetricLearning
# import pytorch_lightning as pl
# from pytorch_lightning.loggers.wandb import WandbLogger
# import torchvision.transforms as transforms
# from datetime import datetime
# from data_utils import cub2011
# from data_utils import stanford_cars
from pytorch_metric_learning import miners, losses
from model.model import KLDistance, KL_d, KLoss, KLLossMetricLearning


if __name__ == "__main__":
    # IMAGE_SIZE = 224
    # BATCH_SIZE = 64

    # train_transform = transforms.Compose([
    #     # transforms.RandomHorizontalFlip(p=0.5),
    #     # transforms.RandomRotation(30),
    #     # transforms.RandomResizedCrop(224, scale=(0.7, 1), ratio=(3/4, 4/3)),
    #     transforms.Resize(IMAGE_SIZE),
    #     transforms.CenterCrop(IMAGE_SIZE),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])

    # transform = transforms.Compose([
    #     transforms.Resize(IMAGE_SIZE),
    #     transforms.CenterCrop(IMAGE_SIZE),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])

    # # cars_trainset = stanford_cars.download_cars('./data', train=True, download=True, transforms=train_transform)
    # # cars_testset = stanford_cars.download_cars('./data', train=True, download=False, transforms=transform)
    # cub_trainset = cub2011.download_cub('./data', train=True, download=True, transforms=train_transform)
    # cub_testset = cub2011.download_cub('./data', train=False, download=False, transforms=transform)

    # # cars_trainloader = torch.utils.data.DataLoader(cars_trainset, batch_size=BATCH_SIZE,
    # #                                           shuffle=True, drop_last=True)
    # # cars_testloader = torch.utils.data.DataLoader(cars_testset, batch_size=BATCH_SIZE,
    # #                                           shuffle=False, drop_last=True)
    # cub_trainloader = torch.utils.data.DataLoader(cub_trainset, batch_size=BATCH_SIZE,
    #                                           shuffle=True, drop_last=True)
    # cub_testloader = torch.utils.data.DataLoader(cub_testset, batch_size=BATCH_SIZE,
    #                                           shuffle=False, drop_last=True)

    # conf_path = path.join("configs", "model_cub.yaml")
    # conf = OmegaConf.load(conf_path)
    # model = KLLossMetricLearning(**conf.get("model"))

    # pl.seed_everything(42)
    # now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # nowname = model.__class__.__name__ + "_" + now
    # logdir = path.join("logs", nowname)
    # ckptdir = path.join(logdir, "checkpoints")
    # default_modelckpt_cfg = {
    #     "dirpath": ckptdir,
    #     "filename": "{epoch:06}",
    #     "verbose": True,
    #     "save_last": True,
    #     "monitor": "val/loss",
    #     "save_top_k": 1,
    #     # "mode": "max",
    # }
    # callbacks = [pl.callbacks.ModelCheckpoint(**default_modelckpt_cfg)]
    # logger = WandbLogger(
    #     project="ZZSN",
    #     name=nowname,
    #     log_model=True,
    #     id=nowname,
    #     # group=str(group),
    #     reinit=True,
    #     allow_val_change=True
    # )
    # trainer = pl.Trainer(logger=logger, callbacks=callbacks, max_epochs=100)
    # trainer.fit(
    #     model, train_dataloaders=cub_trainloader, val_dataloaders=cub_testloader
    # )
    means = torch.Tensor([
        [3, 0, 3],
        [2, 0, 2],
        [7, 3, 2],
        [3, 2, 1]
        ])
    stds = torch.Tensor([
        [0.2, 0.3, 0.1],
        [0.3, 0.1, 2],
        [0.1, 0.1, 0.1],
        [0.2, 0.3, 0.5]
    ])
    labels = torch.Tensor(
        [2, 2, 3, 3]
    )
    # emb = torch.cat((means.unsqueeze(2), stds.unsqueeze(2)), dim=2)
    # m1, s1 = (emb[0].T[0], emb[0].T[1])
    # m2, s2 = (emb[1].T[0], emb[1].T[1])
    # print((m1, s1))
    # print((m2, s2))
    # print(KL_d((m1, s1), (m2, s2)))
    # emb size = [batch size, no. output dims, 2 (for each dim mean and std)]
    # labels = [batch size]


    miner_func = miners.PairMarginMiner(pos_margin=0.2, neg_margin=0.8, distance=KLDistance())
    emb2 = torch.cat((means, stds), dim=1)
    miner_output = miner_func(emb2, labels)
    loss_func = KLoss(distance=KLDistance(), exp_class_distance=0.5, regularization_ratio=0.5, pos_negative_ratio=1)
    print(loss_func.compute_loss(emb2, labels, miner_output, emb2, labels)['loss']['losses'])
    print(loss_func(emb2, labels, miner_output))

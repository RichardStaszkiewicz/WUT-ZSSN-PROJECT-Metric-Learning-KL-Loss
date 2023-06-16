from os import path
import torch
from omegaconf import OmegaConf
from model.model import KLLossMetricLearning
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
import torchvision.transforms as transforms
from datetime import datetime
from data_utils import cub2011
from data_utils import stanford_cars
from pytorch_metric_learning import miners, losses, samplers
from model.model import KLDistance, KL_d, KLoss, KLLossMetricLearning
import torchvision as tv
import wandb


if __name__ == "__main__":
    IMAGE_SIZE = 224
    BATCH_SIZE = 60

    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(30),
        # transforms.RandomResizedCrop(224, scale=(0.7, 1), ratio=(3/4, 4/3)),
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # cars_trainset = stanford_cars.download_cars('./data', train=True, download=True, transforms=train_transform)
    # cars_testset = stanford_cars.download_cars('./data', train=False, download=True, transforms=transform)
    # cub_trainset = cub2011.download_cub('./data', train=True, download=True, transforms=train_transform)
    # cub_testset = cub2011.download_cub('./data', train=False, download=False, transforms=transform)
    # fashion_train = tv.datasets.FashionMNIST("./data", train=True, transform=transforms.ToTensor(), download=True)
    # fashion_test = tv.datasets.FashionMNIST("./data", train=False, transform=transforms.ToTensor(), download=True)
    cifar10_train = tv.datasets.CIFAR10("./data", train=True, transform=transforms.ToTensor(), download=True)
    cifar10_test = tv.datasets.CIFAR10("./data", train=False, transform=transforms.ToTensor(), download=True)

    # cars_trainloader = torch.utils.data.DataLoader(cars_trainset, batch_size=BATCH_SIZE,
    #                                       shuffle=False, drop_last=True, sampler=samplers.MPerClassSampler([label for _, label in cars_trainset.imgs], m=2, length_before_new_iter=len(cars_trainset)))
    # cars_testloader = torch.utils.data.DataLoader(cars_testset, batch_size=BATCH_SIZE,
    #                                       shuffle=False, drop_last=True)
    # cub_trainloader = torch.utils.data.DataLoader(cub_trainset, batch_size=BATCH_SIZE,
    #                                               shuffle=False, drop_last=False, sampler=samplers.MPerClassSampler(cub_trainset.label_list, m=4, length_before_new_iter=len(cub_trainset)))
    # cub_testloader = torch.utils.data.DataLoader(cub_testset, batch_size=BATCH_SIZE,
    #                                              shuffle=False, drop_last=False, sampler=samplers.MPerClassSampler(cub_testset.label_list, m=2, length_before_new_iter=len(cub_testset)))
    # cub_testloader2 = torch.utils.data.DataLoader(cub_testset, batch_size=BATCH_SIZE,
    #                                               shuffle=False, drop_last=False)
    # fashion_trainloader = torch.utils.data.DataLoader(fashion_train, batch_size=BATCH_SIZE,
    #                                                   shuffle=False, drop_last=True, sampler=samplers.MPerClassSampler(fashion_train.targets, m=6, length_before_new_iter=12000))
    # fashion_testloader = torch.utils.data.DataLoader(fashion_test, batch_size=BATCH_SIZE,
    #                                                  shuffle=False, drop_last=True)
    #                                                   shuffle=False, drop_last=True)
    cifar10_trainloader = torch.utils.data.DataLoader(cifar10_train, batch_size=BATCH_SIZE,
                                                      shuffle=False, drop_last=True, sampler=samplers.MPerClassSampler(cifar10_train.targets, m=6, length_before_new_iter=18000))
    cifar10_testloader = torch.utils.data.DataLoader(cifar10_test, batch_size=BATCH_SIZE,
                                                     shuffle=False, drop_last=True, sampler=samplers.MPerClassSampler(cifar10_test.targets, m=6, length_before_new_iter=len(cifar10_test)))
    cifar10_testloader2 = torch.utils.data.DataLoader(cifar10_test, batch_size=BATCH_SIZE,
                                                      shuffle=False, drop_last=True)

    conf_path = path.join("configs", "model_cifar10.yaml")
    conf = OmegaConf.load(conf_path)
    # model = KLLossMetricLearning(**conf.get("model"))

    # ckpt_path = path.join("logs", "KLLossMetricLearning_2023-06-14T12-29-11", "checkpoints", "last.ckpt")
    # state_dict = torch.load(ckpt_path)
    # model.load_state_dict(state_dict["state_dict"])

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
    #     model, train_dataloaders=fashion_trainloader, val_dataloaders=fashion_testloader2
    # )
    # trainer.test(model, fashion_testloader2)

    for exp_dist in [1]:
        for reg_ratio in [0.001, 0.02, 1]:
            for pos_neg in [0.05, 1]:
                conf["model"]["exp_class_distance"] = exp_dist
                conf["model"]["regularization_ratio"] = reg_ratio
                conf["model"]["bh_kwargs"]["pos_neg_ratio"] = pos_neg
                model = KLLossMetricLearning(**conf.get("model"))

                # ckpt_path = path.join("logs", "model1.ckpt")
                # state_dict = torch.load(ckpt_path)
                # model.load_state_dict(state_dict["state_dict"])

                pl.seed_everything(42)
                now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                nowname = model.__class__.__name__ + "_" + now
                logdir = path.join("logs", nowname)
                ckptdir = path.join(logdir, "checkpoints")
                default_modelckpt_cfg = {
                    "dirpath": ckptdir,
                    "filename": "{epoch:06}",
                    "verbose": True,
                    "save_last": True,
                    "monitor": "val/loss",
                    "save_top_k": 1,
                    # "mode": "max",
                }
                callbacks = [pl.callbacks.ModelCheckpoint(**default_modelckpt_cfg)]
                logger = WandbLogger(
                    project="ZZSN",
                    name=nowname,
                    log_model=True,
                    id=nowname,
                    group="CIFAR10-v2",
                    reinit=True,
                    allow_val_change=True
                )
                trainer = pl.Trainer(logger=logger, callbacks=callbacks, max_epochs=10)
                trainer.fit(
                    model, train_dataloaders=cifar10_trainloader, val_dataloaders=cifar10_testloader2
                )
                trainer.test(model, cifar10_testloader2)
                wandb.finish()

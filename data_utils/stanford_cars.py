import os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
import matplotlib
import matplotlib.pyplot as plt
import opendatasets as od
from torch.utils.data import Dataset
matplotlib.rcParams['figure.facecolor'] = '#ffffff'

def find_classes(dir, train=True):
    if train:
        train_classes = os.listdir(dir)
        train_classes.sort()
        train_class_to_idx = {train_classes[i]: i for i in range(len(train_classes))}
        return train_classes, train_class_to_idx
    else:
        test_classes = os.listdir(dir)
        test_classes.sort()
        test_class_to_idx = {test_classes[i]: i for i in range(len(test_classes))}
        return test_classes, test_class_to_idx

def extract_class(Datasets):
  for vals in os.listdir(Datasets):
    print(vals)

def download_cars(dir, train, download, transforms=ToTensor()):
    DATA_DIR_TRAIN = os.path.join(dir, 'stanford-car-dataset-by-classes-folder/car_data/car_data/train')
    DATA_DIR_TEST = os.path.join(dir, 'stanford-car-dataset-by-classes-folder/car_data/car_data/test')

    if download: #and not os.path.isdir(os.path.join(dir, 'stanford-car-dataset-by-classes-folder')):
        dataset_url = 'https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder'
        od.download(dataset_url, data_dir=dir)
    else:
       print(f'folder already exits in given location: {os.path.join(dir, "stanford-car-dataset-by-classes-folder")}')

    train_classes = os.listdir(DATA_DIR_TRAIN)

    test_classes = os.listdir(DATA_DIR_TEST)

    train_classes, train_c_to_idx = find_classes(DATA_DIR_TRAIN, train=True)
    test_classes, test_c_to_idx = find_classes(DATA_DIR_TEST, train=False)

    if train:
       return ImageFolder(DATA_DIR_TRAIN, transform = transforms)
    else:
       return ImageFolder(DATA_DIR_TEST, transform = transforms)


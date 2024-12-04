from __future__ import annotations

import argparse
import copy
import itertools
from collections.abc import Iterator
from functools import partial
from typing import Any
import math

import jax
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
import webdataset as wds
from timm.data.auto_augment import (
    augment_and_mix_transform,
    auto_augment_transform,
    rand_augment_transform,
)
from torch.utils.data import DataLoader, default_collate
from webdataset.shardlists import expand_urls

from utils import read_yaml, preprocess_config

IMAGENET_DEFAULT_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_DEFAULT_STD = np.array([0.229, 0.224, 0.225])




def linear_schedule(epoch, total_epochs, max_syn_ratio=0.7, min_syn_ratio=0.3):
    return max_syn_ratio - (max_syn_ratio - min_syn_ratio) * (epoch / total_epochs)



def cyclic_schedule(epoch, total_epochs, max_syn_ratio=0.7, min_syn_ratio=0.3, cycle_length=10):
    return min_syn_ratio + (max_syn_ratio - min_syn_ratio) * (1 + math.sin(2 * math.pi * (epoch % cycle_length) / cycle_length)) / 2


def step_schedule(epoch, total_epochs):
    if epoch < 0.3 * total_epochs:
        return 0.7  # 70% synthetic
    elif epoch < 0.6 * total_epochs:
        return 0.5  # 50% synthetic
    else:
        return 0.3  # 30% synthetic



import math

def cosine_schedule(epoch, total_epochs, max_syn_ratio=0.7, min_syn_ratio=0.3):
    """
    Cosine decay schedule for adjusting synthetic-to-real ratio.
    Args:
        epoch: Current epoch number.
        total_epochs: Total number of epochs.
        max_syn_ratio: Maximum ratio of synthetic data at the start.
        min_syn_ratio: Minimum ratio of synthetic data at the end.
    Returns:
        Adjusted synthetic data ratio for the current epoch.
    """
    # Corrected formula to ensure the ratio starts at max_syn_ratio and decays to min_syn_ratio
    return max_syn_ratio - (max_syn_ratio - min_syn_ratio) * (1 - math.cos(math.pi * epoch / total_epochs)) / 2




def sigmoid_schedule(epoch, total_epochs, max_syn_ratio=0.7, min_syn_ratio=0.3, k=10, midpoint=None):
    """
    Sigmoid decay schedule for adjusting synthetic-to-real ratio.
    Args:
        epoch: Current epoch number.
        total_epochs: Total number of epochs.
        max_syn_ratio: Maximum ratio of synthetic data at the start.
        min_syn_ratio: Minimum ratio of synthetic data at the end.
        k: Steepness factor for sigmoid curve.
        midpoint: Epoch at which the ratio transitions most quickly. Default is halfway.
    Returns:
        Adjusted synthetic data ratio for the current epoch.
    """
    if midpoint is None:
        midpoint = total_epochs // 2  # Default midpoint at the middle of training

    # Sigmoid decay function (corrected)
    return max_syn_ratio - (max_syn_ratio - min_syn_ratio) / (1 + math.exp(-k * (epoch - midpoint) / total_epochs))







class DynamicMixRatioState:
    def __init__(self, total_batch_size,schedule:str='sigmoid',max_syn_ratio=1.0,min_syn_ratio=0.3):
        self.ratio = max_syn_ratio
        self.total_batch_size = total_batch_size
        self.buffer_syn_x = []
        self.buffer_syn_y = []
        self.buffer_x = []
        self.buffer_y = []

        if schedule=='linear':
            schedule=linear_schedule
        elif schedule=='cosine':
            schedule=cosine_schedule
        elif schedule=='sigmoid':
            schedule=sigmoid_schedule
        else:
            raise NotImplemented()

        self.schedule=partial(schedule,max_syn_ratio=max_syn_ratio,min_syn_ratio=min_syn_ratio)

    def update_mix_ratio(self,epoch,total_epoch):
        self.ratio=self.schedule(epoch,total_epoch,)

    def get_data(self, origin_dataloader_iter, syn_dataloader_iter):
        dataset_mix_ratio = self.ratio
        total_batch_size = self.total_batch_size

        syn_batch_size = int(total_batch_size * dataset_mix_ratio)
        origin_batch_size = total_batch_size - syn_batch_size

        if len(self.buffer_syn_x) < syn_batch_size:
            x, y = next(syn_dataloader_iter)
            self.buffer_syn_x.extend(x)
            self.buffer_syn_y.extend(y)

        if len(self.buffer_x) < origin_batch_size:
            x, y = next(origin_dataloader_iter)
            self.buffer_x.extend(x)
            self.buffer_y.extend(y)

        syn_x, self.buffer_syn_x = self.buffer_syn_x[:syn_batch_size], self.buffer_syn_x[syn_batch_size:]
        syn_y, self.buffer_syn_y = self.buffer_syn_y[:syn_batch_size], self.buffer_syn_y[syn_batch_size:]

        x, self.buffer_x = self.buffer_x[:origin_batch_size], self.buffer_x[origin_batch_size:]
        y, self.buffer_y = self.buffer_y[:origin_batch_size], self.buffer_y[origin_batch_size:]



        x_s=torch.stack([*syn_x,*x])
        y_s=torch.stack([*syn_y,*y])

        # if jax.process_index()==0:
        #     print(x_s.shape,torch.stack(syn_x).shape,torch.stack(x).shape)

        return x_s, y_s

epoch=0
total_epoch=300

state=DynamicMixRatioState(4096)
state.update_mix_ratio(epoch,total_epoch)
import matplotlib.pyplot as plt

x=[]
y=[]

for i in range(epoch,total_epoch):
   x.append(i)
   state.update_mix_ratio(i,total_epoch)
   y.append(state.ratio)

plt.plot(x,y)



state=DynamicMixRatioState(4096,schedule='cosine')
state.update_mix_ratio(epoch,total_epoch,)
import matplotlib.pyplot as plt

x=[]
y=[]

for i in range(epoch,total_epoch):
   x.append(i)
   state.update_mix_ratio(i,total_epoch)
   y.append(state.ratio)

plt.plot(x,y)

state=DynamicMixRatioState(4096,schedule='linear')
state.update_mix_ratio(epoch,total_epoch,)
import matplotlib.pyplot as plt

x=[]
y=[]

for i in range(epoch,total_epoch):
   x.append(i)
   state.update_mix_ratio(i,total_epoch)
   y.append(state.ratio)

plt.plot(x,y)

plt.show()
print(y)


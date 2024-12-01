# Copyright 2024 Jungwoo Park (affjljoo3581)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    def __init__(self, total_batch_size,schedule:str,max_syn_ratio=1.0,min_syn_ratio=0.3):
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

def auto_augment_factory(image_size, auto_augment) -> T.Transform:
    aa_hparams = {
        "translate_const": int(image_size * 0.45),
        "img_mean": tuple((IMAGENET_DEFAULT_MEAN * 0xFF).astype(int)),
    }
    if auto_augment == "none":
        return T.Identity()
    if auto_augment.startswith("rand"):
        return rand_augment_transform(auto_augment, aa_hparams)
    if auto_augment.startswith("augmix"):
        aa_hparams["translate_pct"] = 0.3
        return augment_and_mix_transform(auto_augment, aa_hparams)
    return auto_augment_transform(auto_augment, aa_hparams)


def create_transforms(random_crop,
                      image_size,
                      auto_augment,
                      color_jitter,
                      random_erasing,
                      test_crop_ratio
                      ) -> tuple[nn.Module, nn.Module]:
    if random_crop == "rrc":
        train_transforms = [T.RandomResizedCrop(image_size, interpolation=3)]
    elif random_crop == "src":
        train_transforms = [
            T.Resize(image_size, interpolation=3),
            T.RandomCrop(image_size, padding=4, padding_mode="reflect"),
        ]
    elif random_crop == "none":
        train_transforms = [
            T.Resize(image_size, interpolation=3),
            T.CenterCrop(image_size),
        ]

    train_transforms += [
        T.RandomHorizontalFlip(),
        auto_augment_factory(image_size, auto_augment),
        T.ColorJitter(color_jitter, color_jitter, color_jitter),
        T.RandomErasing(random_erasing, value="random"),
        T.PILToTensor(),
    ]
    valid_transforms = [
        T.Resize(int(image_size / test_crop_ratio), interpolation=3),
        T.CenterCrop(image_size),
        T.PILToTensor(),
    ]
    return T.Compose(train_transforms), T.Compose(valid_transforms)


def repeat_samples(samples: Iterator[Any], repeats: int = 1) -> Iterator[Any]:
    for sample in samples:
        for _ in range(repeats):
            yield copy.deepcopy(sample)


def collate_and_shuffle(batch: list[Any], repeats: int = 1) -> Any:
    return default_collate(sum([batch[i::repeats] for i in range(repeats)], []))


def collate_and_pad(batch: list[Any], batch_size: int = 1) -> Any:
    pad = tuple(torch.full_like(x, fill_value=-1) for x in batch[0])
    return default_collate(batch + [pad] * (batch_size - len(batch)))


def mix_dataloader_iter(train_dataloader, train_origin_dataloader,state:DynamicMixRatioState):
    """
    if jax.process_index() == 0:
        print(f'{train_dataloader=}  {train_origin_dataloader=}', )

    if train_dataloader is None:
        if jax.process_index() == 0:
            print('use origin')

        train_origin_dataloader_iter = iter(train_origin_dataloader)
        while True:
            yield next(train_origin_dataloader_iter)
    elif train_origin_dataloader is None:
        if jax.process_index() == 0:
            print('use generate')
        train_dataloader_iter = iter(train_dataloader)
        while True:
            yield next(train_dataloader_iter)
    else:
        if jax.process_index() == 0:
            print('use generate and origin')
        train_dataloader_iter = iter(train_dataloader)
        train_origin_dataloader_iter = iter(train_origin_dataloader)

        while True:
            yield [torch.cat([x, y], dim=0) for x, y in
                   zip(next(train_dataloader_iter), next(train_origin_dataloader_iter))]
    """
    ratio=state.ratio
    if ratio==1.0:
        if jax.process_index() == 0:
            print('Only use generate')
        train_dataloader_iter = iter(train_dataloader)
        while True:
            yield next(train_dataloader_iter)
    elif ratio==0.0:
        if jax.process_index() == 0:
            print('Only use origin')

        train_origin_dataloader_iter = iter(train_origin_dataloader)
        while True:
            yield next(train_origin_dataloader_iter)
    else:

        if jax.process_index() == 0:
            print('use generate and origin')
        train_dataloader_iter = iter(train_dataloader)
        train_origin_dataloader_iter = iter(train_origin_dataloader)
        while True:
            x,y=state.get_data(origin_dataloader_iter=train_origin_dataloader_iter,syn_dataloader_iter=train_dataloader_iter)
            yield x,y




def create_dataloaders(
        train_dataset_shards,
        valid_dataset_shards,
        train_batch_size,
        valid_batch_size,
        train_loader_workers,
        valid_loader_workers,
        augment_repeats,
        shuffle_seed,
        random_crop,
        image_size,
        auto_augment,
        color_jitter,
        random_erasing,
        auto_generated_augment,
        color_generated_jitter,
        random_generated_erasing,
        test_crop_ratio,
        generated_dataset_shards,
        grad_accum=1,
        dataset_mix_ratio=0.8,
        max_syn_ratio=1.0,
        min_syn_ratio=0.3,
        scheduler='linear'

):


    train_dataloader, train_origin_dataloader, valid_dataloader = None, None, None
    train_transform, valid_transform = create_transforms(random_crop,
                                                         image_size,
                                                         auto_augment,
                                                         color_jitter,
                                                         random_erasing,
                                                         test_crop_ratio
                                                         )

    train_generated_transform, valid_transform = create_transforms(random_crop,
                                                               image_size,
                                                               auto_generated_augment,
                                                               color_generated_jitter,
                                                               random_generated_erasing,
                                                               test_crop_ratio
                                                               )
    total_batch_size = train_batch_size // jax.process_count() //grad_accum
    train_batch_size = int(total_batch_size * dataset_mix_ratio)
    train_origin_batch_size = total_batch_size - train_batch_size

    state=DynamicMixRatioState(total_batch_size,scheduler,max_syn_ratio,min_syn_ratio)


    generated_train_loader_workers = 10

    files = []
    for url in generated_dataset_shards:
        files.extend(expand_urls(url))
    generated_dataset_shards = files
    # generated_dataset_shards = 'gs://shadow-center-2b/imagenet-generated-100steps-annotated/shards-{00000..01500}.tar'


    dataset = wds.DataPipeline(
        wds.SimpleShardList(train_dataset_shards, seed=shuffle_seed),
        itertools.cycle,
        wds.detshuffle(),
        wds.slice(jax.process_index(), None, jax.process_count()),
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=wds.ignore_and_continue),
        wds.detshuffle(),
        wds.decode("pil", handler=wds.ignore_and_continue),
        wds.to_tuple("jpg", "cls", handler=wds.ignore_and_continue),
        partial(repeat_samples, repeats=augment_repeats),
        wds.map_tuple(train_transform, torch.tensor),
    )
    train_origin_dataloader = DataLoader(
        dataset,
        batch_size=total_batch_size,
        num_workers=train_loader_workers,
        collate_fn=partial(collate_and_shuffle, repeats=augment_repeats),
        drop_last=True,
        prefetch_factor=20,
        persistent_workers=True,
    )


    dataset = wds.DataPipeline(
        wds.SimpleShardList(generated_dataset_shards, seed=shuffle_seed),
        itertools.cycle,
        wds.detshuffle(),
        wds.slice(jax.process_index(), None, jax.process_count()),
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=wds.ignore_and_continue),
        wds.detshuffle(),
        wds.decode("pil", handler=wds.ignore_and_continue),
        wds.to_tuple("jpg", "cls", handler=wds.ignore_and_continue),
        partial(repeat_samples, repeats=augment_repeats),
        wds.map_tuple(train_generated_transform, torch.tensor),
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=total_batch_size,
        num_workers=generated_train_loader_workers,
        collate_fn=partial(collate_and_shuffle, repeats=augment_repeats),
        drop_last=True,
        prefetch_factor=20,
        persistent_workers=True,
    )

    if valid_dataset_shards is not None:
        dataset = wds.DataPipeline(
            wds.SimpleShardList(valid_dataset_shards),
            wds.slice(jax.process_index(), None, jax.process_count()),
            wds.split_by_worker,
            # wds.cached_tarfile_to_samples(),
            wds.tarfile_to_samples(handler=wds.ignore_and_continue),
            wds.decode("pil"),
            wds.to_tuple("jpg", "cls"),
            wds.map_tuple(valid_transform, torch.tensor),
        )
        valid_dataloader = DataLoader(
            dataset,
            batch_size=(batch_size := valid_batch_size // jax.process_count()),
            num_workers=valid_loader_workers,
            collate_fn=partial(collate_and_pad, batch_size=batch_size),
            drop_last=False,
            prefetch_factor=20,
            persistent_workers=True,
        )
    return mix_dataloader_iter(train_dataloader, train_origin_dataloader,state), valid_dataloader,state
    # return train_dataloader, valid_dataloader


if __name__ == "__main__":
    yaml = read_yaml('configs/test.yaml')
    yaml = preprocess_config(yaml)
    train_dataloader, valid_dataloader = create_dataloaders(**yaml['dataset'])

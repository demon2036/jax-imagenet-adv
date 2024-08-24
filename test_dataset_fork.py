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

from utils import read_yaml, preprocess_config

IMAGENET_DEFAULT_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_DEFAULT_STD = np.array([0.229, 0.224, 0.225])


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


def mix_dataloader_iter(train_dataloader, train_origin_dataloader):
    if jax.process_index() == 0:
        print(train_dataloader, train_origin_dataloader)

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
        test_crop_ratio,
        grad_accum=1,
):
    # print(train_dataset_shards,
    #       valid_dataset_shards,
    #       train_batch_size,
    #       valid_batch_size,
    #       train_loader_workers,
    #       valid_loader_workers,
    #       augment_repeats,
    #       shuffle_seed,
    #       random_crop,
    #       image_size,
    #       auto_augment,
    #       color_jitter,
    #       random_erasing,
    #       test_crop_ratio,
    #       grad_accum)

    train_dataloader, train_origin_dataloader, valid_dataloader = None, None, None
    train_transform, valid_transform = create_transforms(random_crop,
                                                         image_size,
                                                         auto_augment,
                                                         color_jitter,
                                                         random_erasing,
                                                         test_crop_ratio
                                                         )
    dataset_mix_ratio = 0.2
    total_batch_size = train_batch_size // jax.process_count()
    train_batch_size = int(total_batch_size * dataset_mix_ratio)
    train_origin_batch_size = total_batch_size - train_batch_size
    generated_dataset_shards = 'gs://shadow-center-2b/imagenet-generated-100steps/shards-{00000..06399}.tar'
    # generated_dataset_shards = 'gs://shadow-center-2b/imagenet-generated-100steps-annotated/shards-{00000..01500}.tar'

    if train_origin_batch_size is not None:
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
            batch_size=train_origin_batch_size,
            num_workers=train_loader_workers,
            collate_fn=partial(collate_and_shuffle, repeats=augment_repeats),
            drop_last=True,
            prefetch_factor=20,
            persistent_workers=True,
        )

    if train_batch_size > 0:
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
            wds.map_tuple(train_transform, torch.tensor),
        )
        train_dataloader = DataLoader(
            dataset,
            batch_size=train_batch_size,
            num_workers=train_loader_workers,
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
    return mix_dataloader_iter(train_dataloader, train_origin_dataloader), valid_dataloader
    # return train_dataloader, valid_dataloader


if __name__ == "__main__":
    yaml = read_yaml('configs/test.yaml')
    yaml = preprocess_config(yaml)
    train_dataloader, valid_dataloader = create_dataloaders(**yaml['dataset'])

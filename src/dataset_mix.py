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
from typing import Any, Tuple

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
from torchvision.transforms.v2 import Compose

IMAGENET_DEFAULT_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_DEFAULT_STD = np.array([0.229, 0.224, 0.225])


def auto_augment_factory(args: argparse.Namespace) -> T.Transform:
    aa_hparams = {
        "translate_const": int(args.image_size * 0.45),
        "img_mean": tuple((IMAGENET_DEFAULT_MEAN * 0xFF).astype(int)),
    }
    if args.auto_augment == "none":
        return T.Identity()
    if args.auto_augment.startswith("rand"):
        return rand_augment_transform(args.auto_augment, aa_hparams)
    if args.auto_augment.startswith("augmix"):
        aa_hparams["translate_pct"] = 0.3
        return augment_and_mix_transform(args.auto_augment, aa_hparams)
    return auto_augment_transform(args.auto_augment, aa_hparams)


def create_transforms(args: argparse.Namespace) -> tuple[Compose, Compose, Compose]:
    if args.random_crop == "rrc":
        train_transforms = [T.RandomResizedCrop(args.image_size, interpolation=3)]
    elif args.random_crop == "src":
        train_transforms = [
            T.Resize(args.image_size, interpolation=3),
            T.RandomCrop(args.image_size, padding=4, padding_mode="reflect"),
        ]
    elif args.random_crop == "none":
        train_transforms = [
            T.Resize(args.image_size, interpolation=3),
            T.CenterCrop(args.image_size),
        ]

    train_generated_transforms = [*train_transforms, T.RandomHorizontalFlip(), T.PILToTensor(), ]

    train_transforms += [
        T.RandomHorizontalFlip(),
        auto_augment_factory(args),
        T.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter),
        T.RandomErasing(args.random_erasing, value="random"),
        T.PILToTensor(),
    ]
    valid_transforms = [
        T.Resize(int(args.image_size / args.test_crop_ratio), interpolation=3),
        T.CenterCrop(args.image_size),
        T.PILToTensor(),
    ]
    return T.Compose(train_transforms),T.Compose(train_generated_transforms), T.Compose(valid_transforms)


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
        args: argparse.Namespace,
) -> tuple[Any | None, DataLoader | None]:
    train_dataloader, valid_dataloader, train_origin_dataloader = None, None, None
    train_transform,train_generated_transforms, valid_transform = create_transforms(args)

    dataset_mix_ratio = 0.0
    total_batch_size = args.train_batch_size // jax.process_count()
    train_batch_size = int(total_batch_size * dataset_mix_ratio)
    train_origin_batch_size = total_batch_size - train_batch_size

    args.generated_dataset_shards = 'gs://shadow-center-2b/imagenet-generated-100steps/shards-{00000..06399}.tar'

    if args.train_dataset_shards is not None:
        if train_batch_size > 0:
            dataset = wds.DataPipeline(
                wds.SimpleShardList(args.generated_dataset_shards, seed=args.shuffle_seed),
                itertools.cycle,
                wds.detshuffle(),
                wds.slice(jax.process_index(), None, jax.process_count()),
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=wds.warn_and_stop),
                wds.detshuffle(),
                wds.decode("pil", handler=wds.warn_and_stop),
                wds.to_tuple("jpg", "cls", handler=wds.warn_and_stop),
                partial(repeat_samples, repeats=args.augment_repeats),
                wds.map_tuple(train_generated_transforms, torch.tensor),
            )
            train_dataloader = DataLoader(
                dataset,
                batch_size=train_batch_size // args.grad_accum,
                num_workers=args.train_loader_workers,
                collate_fn=partial(collate_and_shuffle, repeats=args.augment_repeats),
                drop_last=True,
                prefetch_factor=20,
                persistent_workers=True,
            )
        else:
            train_dataloader = None

        if train_origin_batch_size > 0:
            dataset = wds.DataPipeline(
                wds.SimpleShardList(args.train_dataset_shards, seed=args.shuffle_seed),
                itertools.cycle,
                wds.detshuffle(seed=1),
                wds.slice(jax.process_index(), None, jax.process_count()),
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=wds.warn_and_stop),  #handler=wds.ignore_and_continue
                wds.detshuffle(),
                wds.decode("pil", handler=wds.warn_and_stop),
                wds.to_tuple("jpg", "cls", handler=wds.warn_and_stop),
                partial(repeat_samples, repeats=args.augment_repeats),
                wds.map_tuple(train_transform, torch.tensor),
            )
            train_origin_dataloader = DataLoader(
                dataset,
                batch_size=train_origin_batch_size // args.grad_accum,
                num_workers=args.train_loader_workers,
                collate_fn=partial(collate_and_shuffle, repeats=args.augment_repeats),
                drop_last=True,
                prefetch_factor=20,
                persistent_workers=True,
            )
        else:
            train_origin_dataloader = None

    if args.valid_dataset_shards is not None:
        dataset = wds.DataPipeline(
            wds.SimpleShardList(args.valid_dataset_shards),
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
            batch_size=(batch_size := args.valid_batch_size // jax.process_count()),
            num_workers=args.valid_loader_workers,
            collate_fn=partial(collate_and_pad, batch_size=batch_size),
            drop_last=False,
            prefetch_factor=20,
            persistent_workers=True,
        )

    return mix_dataloader_iter(train_dataloader, train_origin_dataloader), valid_dataloader
    # return train_dataloader, valid_dataloader

import functools
from typing import Any, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as tnn
import einops

from dataset import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

Dense = functools.partial(nn.Dense, kernel_init=nn.initializers.truncated_normal(0.02))
Conv = functools.partial(nn.Conv, kernel_init=nn.initializers.truncated_normal(0.02))


class Identity(nn.Module):
    def __call__(self, x):
        return x


class Mlp(nn.Module):
    hidden_features: int
    out_features: int
    act_layer: Any = functools.partial(nn.gelu, approximate=True)

    def setup(self) -> None:
        self.fc1 = Dense(self.hidden_features, )
        self.fc2 = Dense(self.out_features, )

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.fc2(x)
        return x


class ConvNeXtBlock(nn.Module):
    in_channels: int
    out_channels: int
    # kernel_size:int=7
    stride: int = 1
    ls_init_value: float = 1e-6
    drop_path_rate: float = 0.0

    def setup(self) -> None:
        self.conv_dw = Conv(self.out_channels, (7, 7),
                            # use_bias=use_bias,
                            feature_group_count=self.in_channels,
                            # precision='highest',
                            padding=[(3, 3), (3, 3)], )

        self.norm = nn.LayerNorm(epsilon=1e-6, use_fast_variance=True)
        self.mlp = Mlp(self.out_channels * 4, self.out_channels)

        if self.ls_init_value is not None:
            self.gamma = self.param("gamma", nn.initializers.constant(self.ls_init_value), (self.out_channels,))
        else:
            self.gamma = None

        self.shortcut = Identity()

        self.drop_path = nn.Dropout(self.drop_path_rate, broadcast_dims=(1, 2, 3))

        # self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

    def __call__(self, x, det=True):
        shortcut = x
        x = self.conv_dw(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = self.shortcut(shortcut) + self.drop_path(x, det)
        return x


class ConvNeXtStage(nn.Module):
    in_channels: int
    out_channels: int
    # kernel_size:int=7
    stride: int = 1
    depth: int = 2
    drop_path_rates: Tuple[float] = None

    def setup(self) -> None:
        if self.in_channels != self.out_channels or self.stride > 1:
            ds_ks = 2
            self.downsample = nn.Sequential([
                nn.LayerNorm(epsilon=1e-6, use_fast_variance=True),
                Conv(self.out_channels, kernel_size=(2, 2), strides=(2, 2), padding=(0, 0))
            ])
            in_chs = self.out_channels
        else:
            self.downsample = Identity()
            in_chs = self.in_channels

        stage_blocks = []

        for i in range(self.depth):
            stage_blocks.append(
                ConvNeXtBlock(in_chs, self.out_channels,drop_path_rate=self.drop_path_rates[i])
            )

            in_chs = self.out_channels

        # self.blocks = nn.Sequential(stage_blocks)
        self.blocks = stage_blocks

    def __call__(self, x,det=True):

        x = self.downsample(x)

        for block in self.blocks:
            x = block(x,det)
        return x


class ConvNeXt(nn.Module):
    # out_channels: int
    # kernel_size:int=7
    # stride: int = 1
    depths: Tuple[int, ...] = (3, 3, 9, 3)
    dims: Tuple[int, ...] = (96, 192, 384, 768)
    num_classed: int = 1000
    labels: int | None = 1000
    drop_path_rate:float =0.1

    def setup(self) -> None:
        self.stem = nn.Sequential([
            Conv(self.dims[0], kernel_size=(4, 4), strides=(4, 4)),
            nn.LayerNorm(epsilon=1e-6, use_fast_variance=True),
        ])

        stages = []
        prev_chs = self.dims[0]
        dp_rates = [x.tolist() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths)).split(self.depths)]
        for i in range(4):
            stride = 2 if i > 0 else 1

            out_chs = self.dims[i]

            stages.append(
                ConvNeXtStage(
                    prev_chs,
                    out_chs,
                    stride=stride,
                    depth=self.depths[i],
                    drop_path_rates=dp_rates[i],
                )
            )

            prev_chs = out_chs

        self.stages = stages

        self.norm = nn.LayerNorm(epsilon=1e-6, use_fast_variance=True)
        self.head = Dense(self.num_classed)

    def __call__(self, x, det=True):
        x = (x - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x,det)
        x = jnp.mean(x, axis=[1, 2])
        x = self.norm(x)
        x = self.head(x)
        return x

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







class Identity(nn.Module):
    def __call__(self, x):
        return x


class Mlp(nn.Module):
    hidden_features: int
    out_features: int
    act_layer: Any = functools.partial(nn.gelu, approximate=True)

    def setup(self) -> None:
        self.fc1 = nn.Dense(self.hidden_features, )
        self.fc2 = nn.Dense(self.out_features, )

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

    def setup(self) -> None:
        self.conv_dw = nn.Conv(self.out_channels, (7, 7),
                               # use_bias=use_bias,
                               feature_group_count=self.in_channels,
                               # precision='highest',
                               padding=[(3, 3), (3, 3)], )

        self.norm = nn.LayerNorm(epsilon=1e-6, use_fast_variance=True)
        self.mlp = Mlp(self.out_channels * 4, self.out_channels)

        # if self.stride!=1 or self.in_channels!=self.out_channels:
        #     self.shortcut=
        # else:
        self.shortcut = Identity()

    def __call__(self, x, *args, **kwargs):
        shortcut = x
        x = self.conv_dw(x)
        x = self.norm(x)
        x = self.mlp(x)
        x = self.shortcut(shortcut) + x
        return x


class ConvNeXtStage(nn.Module):
    in_channels: int
    out_channels: int
    # kernel_size:int=7
    stride: int = 1
    depth: int = 2

    def setup(self) -> None:
        # print(self.in_channels)
        if self.in_channels != self.out_channels or self.stride > 1:
            ds_ks = 2
            self.downsample = nn.Sequential([
                nn.LayerNorm(epsilon=1e-6, use_fast_variance=True),
                nn.Conv(self.out_channels, kernel_size=(2, 2), strides=(2, 2), padding=(0, 0))
            ])
            in_chs = self.out_channels
        else:
            self.downsample = Identity()
            in_chs = self.in_channels

        stage_blocks = []

        for i in range(self.depth):
            stage_blocks.append(
                ConvNeXtBlock(in_chs, self.out_channels)
            )

            in_chs = self.out_channels

        # self.blocks = nn.Sequential(stage_blocks)
        self.blocks = stage_blocks

    def __call__(self, x):

        x = self.downsample(x)

        for block in self.blocks:
            x = block(x)
        return x


class ConvNeXt(nn.Module):
    # out_channels: int
    # kernel_size:int=7
    # stride: int = 1
    depths: Tuple[int, ...] = (3, 3, 9, 3)
    dims: Tuple[int, ...] = (96, 192, 384, 768)
    num_classed: int = 1000
    labels: int | None = 1000

    def setup(self) -> None:
        self.stem = nn.Sequential([
            nn.Conv(self.dims[0], kernel_size=(4, 4), strides=(4, 4)),
            nn.LayerNorm(epsilon=1e-6, use_fast_variance=True),
        ])

        stages = []
        print(self.depths)

        prev_chs = self.dims[0]
        for i in range(4):
            stride = 2 if i > 0 else 1

            out_chs = self.dims[i]

            stages.append(
                ConvNeXtStage(
                    prev_chs,
                    out_chs,
                    stride=stride,
                    depth=self.depths[i]
                )
            )

            prev_chs = out_chs

        self.stages = stages

        self.norm = nn.LayerNorm(epsilon=1e-6, use_fast_variance=True)
        self.head = nn.Dense(self.num_classed)

    def __call__(self, x,det=True):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = jnp.mean(x, axis=[1, 2])
        x = self.norm(x)
        x = self.head(x)
        return x



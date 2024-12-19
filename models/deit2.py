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

from dataclasses import dataclass, fields
from functools import partial
from typing import Any, Literal, Callable

import einops
import flax.linen as nn
import flax.linen.initializers as init
import jax.experimental.pallas.ops.tpu.flash_attention
import jax.numpy as jnp
from chex import Array
from jax import NamedSharding

from utils import get_jax_mesh2


# from datasets import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from utils2 import fixed_sincos2d_embeddings




def fixed_sincos2d_embeddings(ncols: int, nrows: int, dim: int) -> Array:
    freqs = 1 / (10000 ** jnp.linspace(0, 1, dim // 4))
    x = jnp.outer(jnp.arange(0, nrows, dtype=jnp.float32), freqs)
    y = jnp.outer(jnp.arange(0, ncols, dtype=jnp.float32), freqs)

    x = jnp.broadcast_to(x[None, :, :], (ncols, nrows, dim // 4))
    y = jnp.broadcast_to(y[:, None, :], (ncols, nrows, dim // 4))
    return jnp.concatenate((jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)), axis=2)



DenseGeneral = partial(nn.DenseGeneral, kernel_init=init.truncated_normal(0.02))
Dense = partial(nn.Dense, kernel_init=init.truncated_normal(0.02))
Conv = partial(nn.Conv, kernel_init=init.truncated_normal(0.02))


@dataclass
class ViTBase:
    layers: int = 12
    dim: int = 768
    heads: int = 12
    labels: int | None = 1000
    layerscale: bool = False

    patch_size: int = 16
    image_size: int = 224
    posemb: Literal["learnable", "sincos2d"] = "sincos2d"
    pooling: Literal["cls", "gap"] = "cls"
    qk_norm: bool = False
    use_fc_norm: bool = True
    reduce_include_prefix: bool = False

    dropout: float = 0.0
    droppath: float = 0.0
    grad_ckpt: bool = False
    use_kan: bool = False
    polynomial_degree: int = 8

    @property
    def kwargs(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(ViTBase)}

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @property
    def hidden_dim(self) -> int:
        return 4 * self.dim

    @property
    def num_patches(self) -> tuple[int, int]:
        return (self.image_size // self.patch_size,) * 2


class PatchEmbed(ViTBase, nn.Module):
    def setup(self):
        self.wte = Conv(
            self.dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            use_bias=True
        )
        # if self.pooling == "cls":
        self.cls_token = self.param(
            "cls_token", init.truncated_normal(0.02), (1, 1, self.dim)
        )

        if self.posemb == "learnable":
            self.wpe = self.param(
                "wpe", init.truncated_normal(0.02), (*self.num_patches, self.dim)
            )
        elif self.posemb == "sincos2d":
            self.wpe = fixed_sincos2d_embeddings(*self.num_patches, self.dim)

    def __call__(self, x: Array) -> Array:
        x = (self.wte(x) ).reshape(x.shape[0], -1, self.dim)

        # x = (self.wte(x) + self.wpe).reshape(x.shape[0], -1, self.dim)
        # if self.pooling == "cls":
        #     cls_token = jnp.repeat(self.cls_token, x.shape[0], axis=0)
        #     x = jnp.concatenate((cls_token, x), axis=1)
        return x


class Identity(nn.Module):
    def __call__(self, x):
        return x

#
# class Attention(ViTBase, nn.Module):
#     def setup(self):
#         self.q_norm = nn.LayerNorm() if self.qk_norm else Identity()
#         self.k_norm = nn.LayerNorm() if self.qk_norm else Identity()
#         self.wq = DenseGeneral((self.heads, self.head_dim))
#         self.wk = DenseGeneral((self.heads, self.head_dim))
#         self.wv = DenseGeneral((self.heads, self.head_dim))
#         self.wo = DenseGeneral(self.dim, axis=(-2, -1))
#         self.drop = nn.Dropout(self.dropout)
#
#     def __call__(self, x: Array, det: bool = True) -> Array:
#         z = jnp.einsum("bqhd,bkhd->bhqk", self.q_norm(self.wq(x)) / self.head_dim ** 0.5, self.k_norm(self.wk(x)))
#         z = jnp.einsum("bhqk,bkhd->bqhd", self.drop(nn.softmax(z), det), self.wv(x))
#         return self.drop(self.wo(z), det)






class Attention(ViTBase, nn.Module):
    def setup(self):
        self.q_norm = nn.LayerNorm() if self.qk_norm else Identity()
        self.k_norm = nn.LayerNorm() if self.qk_norm else Identity()
        self.wq = Dense(self.dim)
        self.wk = Dense(self.dim)
        self.wv = Dense(self.dim)
        self.wo = Dense(self.dim)
        self.drop = nn.Dropout(self.dropout)

    def __call__(self, x: Array, det: bool = True) -> Array:


        q=self.wq(x)
        k=self.wk(x)
        v=self.wv(x)

        q=einops.rearrange(q,'b n (h d)-> b h n d',h=self.heads)
        k = einops.rearrange(k, 'b n (h d)-> b h n d',h=self.heads)
        v = einops.rearrange(v, 'b n (h d)-> b h n d',h=self.heads)
        # jnp.array().swapaxes()
        z=(q@k.swapaxes(-2,-1))/self.head_dim**0.5
        z=nn.softmax(z)
        z=z@v

        # z=jax.experimental.pallas.ops.tpu.flash_attention.flash_attention(q,k,v)

        z=einops.rearrange(z,'b h n d -> b n (h d)')
        return self.wo(z)

        # z = jnp.einsum("bqhd,bkhd->bhqk", self.q_norm(self.wq(x)) / self.head_dim ** 0.5, self.k_norm(self.wk(x)))
        # z = jnp.einsum("bhqk,bkhd->bqhd", self.drop(nn.softmax(z), det), self.wv(x))
        # return self.drop(self.wo(z), det)





class FeedForward(ViTBase, nn.Module):
    dense_init: Callable = nn.initializers.xavier_normal()
    def setup(self):
        self.w1 = Dense(self.hidden_dim)
        self.w2 = Dense(self.dim,
                        kernel_init=nn.with_logical_partitioning(self.dense_init, ('mlp', 'embed')),

                        use_bias=False,)
        self.drop = nn.Dropout(self.dropout)

    def __call__(self, x: Array, det: bool = True) -> Array:
        return self.drop(self.w2(self.drop(nn.gelu(self.w1(x)), det)), det)


class ViTLayer(ViTBase, nn.Module):
    def setup(self):
        self.attn = Attention(**self.kwargs)
        self.ff = FeedForward(**self.kwargs)

        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.drop = nn.Dropout(self.droppath, broadcast_dims=(1, 2))

        self.scale1 = self.scale2 = 1.0
        if self.layerscale:
            self.scale1 = self.param("scale1", init.constant(1e-4), (self.dim,))
            self.scale2 = self.param("scale2", init.constant(1e-4), (self.dim,))
            # self.scale1 = self.param("scale1", init.constant(1e-6), (self.dim,))
            # self.scale2 = self.param("scale2", init.constant(1e-6), (self.dim,))

    def __call__(self, x: Array, det: bool = True) -> Array:
        # x = x + self.drop(self.scale1 * self.attn(self.norm1(x), det), det)
        # x = x + self.drop(self.scale2 * self.ff(self.norm2(x), det), det)



        x=self.ff.w2(x)
        return x


# mesh_dim_m = '-1,1,4'
# # mesh_dim = '1,1,-1'
# mesh_m = get_jax_mesh2(mesh_dim_m)
# sharding_m = jax.sharding.NamedSharding(
#     mesh_m, jax.sharding.PartitionSpec("dp",None,'mp' ))

class ViT(ViTBase, nn.Module):
    def setup(self):
        self.embed = PatchEmbed(**self.kwargs)
        self.drop = nn.Dropout(self.dropout)

        # The layer class should be wrapped with `nn.remat` if `grad_ckpt` is enabled.
        layer_fn = nn.remat(ViTLayer) if self.grad_ckpt else ViTLayer
        self.layer = [layer_fn(**self.kwargs) for _ in range(self.layers)]

        # self.norm = nn.LayerNorm()

        self.norm = nn.LayerNorm() if not self.use_fc_norm else Identity()
        self.fc_norm = nn.LayerNorm() if self.use_fc_norm else Identity()

        # print(self.norm, self.fc_norm)

        self.head = Dense(self.labels) if self.labels is not None else None
        self.pre_norm=nn.LayerNorm()

    def __call__(self, x: Array, det: bool = True) -> Array:
        # x = (x - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD

        # if isinstance(x,jax._src.interpreters.partial_eval.DynamicJaxprTracer):
        #     # jax.debug.visualize_array_sharding(x[0])
        #     print(x.shape)
        #     jax.debug.inspect_array_sharding(x,callback=print)
        # else:
        #     print(type(x))

        x = self.drop(self.embed(x), det)
        x=jax.lax.stop_gradient(x)

        # x=self.pre_norm(x)
        # x = jax.lax.with_sharding_constraint(x, sharding_m)
        for layer in self.layer:
            x=nn.with_logical_constraint(x,('batch','vocab','embed'))
            x = layer(x, det)
            # x = jax.lax.with_sharding_constraint(x, sharding_m)
            # if isinstance(x,jax._src.interpreters.ad.JVPTracer):
            #     # jax.debug.visualize_array_sharding(x[0])
            #     print(x.shape)
            #     jax.debug.inspect_array_sharding(x,callback=print)


            # x = jax.lax.with_sharding_constraint(x, sharding_m)

        # x = self.norm(x)
        # x=jax.lax.with_sharding_constraint(x,sharding_m)

        #
        # if isinstance(x,jax._src.interpreters.ad.JVPTracer):
        #     # jax.debug.visualize_array_sharding(x[0])
        #     print(x.shape)
        #     jax.debug.inspect_array_sharding(x,callback=print)

        # If the classification head is not defined, then return the output of all
        # tokens instead of pooling to a single vector and then calculate class logits.
        if self.head is None:
            return x
        """
        if self.pooling == "cls":
            x = x[:, 0, :]
        elif self.pooling == "gap":
            x = x[:, 0:].mean(1)
        return self.head(x)
        """
        # jnp.array().sharding
        # print(x.sharding)

        #     print(x.sharding)
        # print(type(x),)
        if self.pooling == "cls":
            x = x[:, 0, :]
        elif self.pooling == "gap":
            x = x if self.reduce_include_prefix else x[:, 1:]
            x = x.mean(1)
        else:
            raise NotImplemented()

        # x = self.fc_norm(x)
        # x=jax.lax.with_sharding_constraint(x,NamedSharding(mesh,jax.sharding.PartitionSpec('dp','mp',)))
        return self.head(x)


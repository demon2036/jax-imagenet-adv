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
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax
from chex import ArrayTree, PRNGKey
from flax.training import train_state
from flax.training.common_utils import shard_prng_key


class TrainState(train_state.TrainState):
    mixup_rng: PRNGKey
    dropout_rng: PRNGKey
    adv_rng: PRNGKey

    micro_step: int = 0
    micro_in_mini: int = 1
    grad_accum: ArrayTree | None = None

    ema_params: Any = None
    ema_decay: float = 0.9998

    def split_rngs(self) -> tuple[ArrayTree, ArrayTree]:
        mixup_rng, new_mixup_rng = jax.random.split(self.mixup_rng)
        dropout_rng, new_dropout_rng = jax.random.split(self.dropout_rng)
        adv_rng, new_adv_rng = jax.random.split(self.adv_rng)

        rngs = {"mixup": mixup_rng, "dropout": dropout_rng, 'adv': adv_rng}
        updates = {"mixup_rng": new_mixup_rng, "dropout_rng": new_dropout_rng, 'adv_rng': new_adv_rng}
        return rngs, updates

    def replicate(self) -> TrainState:
        return flax.jax_utils.replicate(self).replace(
            mixup_rng=shard_prng_key(self.mixup_rng),
            dropout_rng=shard_prng_key(self.dropout_rng),
        )


@partial(jax.pmap, axis_name="batch", donate_argnums=0)
def training_step(state: TrainState, batch: ArrayTree) -> tuple[TrainState, ArrayTree]:
    def loss_fn(params: ArrayTree) -> ArrayTree:
        use_pgd=False
        metrics = state.apply_fn({"params": params}, *batch, det=False, rngs=rngs,use_trade=not use_pgd,use_pgd=use_pgd,)
        metrics = jax.tree_map(jnp.mean, metrics)
        return metrics["loss"], metrics

    def update_fn(state: TrainState) -> TrainState:
        # Collect a global gradient from the accumulated gradients and apply actual
        # parameter update with resetting the accumulations to zero.
        grads = jax.tree_map(lambda g: g / state.micro_in_mini, state.grad_accum)
        return state.apply_gradients(
            grads=jax.lax.pmean(grads, axis_name="batch"),
            grad_accum=jax.tree_map(jnp.zeros_like, state.grad_accum),
            micro_step=state.micro_step % state.micro_in_mini,
        )

    rngs, updates = state.split_rngs()
    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    metrics = jax.lax.pmean(metrics, axis_name="batch")

    # Update parameters with the gradients. If the gradient accumulation is enabled,
    # then the parameters will be updated at the end of each mini-batch step. In every
    # micro steps, the gradients will be accumulated.
    if state.grad_accum is None:
        state = state.apply_gradients(grads=jax.lax.pmean(grads, axis_name="batch"))
    else:
        state = state.replace(
            grad_accum=jax.tree_map(lambda ga, g: ga + g, state.grad_accum, grads),
            micro_step=state.micro_step + 1,
        )
        state = jax.lax.cond(
            state.micro_step == state.micro_in_mini, update_fn, lambda x: x, state
        )

    # if state.ema_params is not None:

    new_ema_params = jax.tree_util.tree_map(
        lambda ema, normal: ema * state.ema_decay + (1 - state.ema_decay) * normal,
        state.ema_params, state.params)
    state = state.replace(ema_params=new_ema_params)

    return state.replace(**updates), metrics | state.opt_state.hyperparams


@partial(jax.pmap, axis_name="batch")
def validation_step(state: TrainState, batch: ArrayTree) -> ArrayTree:
    metrics = state.apply_fn(
        {"params": state.ema_params},
        images=batch[0],
        labels=jnp.where(batch[1] != -1, batch[1], 0),
        det=True,
    )
    metrics["num_samples"] = batch[1] != -1
    metrics = jax.tree_map(lambda x: (x * (batch[1] != -1)).sum(), metrics)
    return jax.lax.psum(metrics, axis_name="batch")


@partial(jax.pmap, axis_name="batch")
def validation_adv_step(state: TrainState, batch: ArrayTree) -> ArrayTree:
    rngs, updates = state.split_rngs()
    metrics = state.apply_fn(
        {"params": state.ema_params},
        images=batch[0],
        labels=jnp.where(batch[1] != -1, batch[1], 0),
        det=True, use_pgd=False
    )

    metrics_adv = state.apply_fn(
        {"params": state.ema_params},
        images=batch[0],
        labels=jnp.where(batch[1] != -1, batch[1], 0),
        det=True, use_pgd=True,rngs=rngs,
    )

    metrics_adv = {'adv' + k: v for k, v in metrics_adv.items()}
    metrics.update(metrics_adv)

    metrics["num_samples"] = batch[1] != -1
    metrics = jax.tree_util.tree_map(lambda x: (x * (batch[1] != -1)).sum(), metrics)
    return jax.lax.psum(metrics, axis_name="batch")

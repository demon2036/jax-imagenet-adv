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

from utils import get_jax_mesh2


class TrainState(train_state.TrainState):
    mixup_rng: PRNGKey
    dropout_rng: PRNGKey
    adv_rng: PRNGKey

    micro_step: int = 0
    micro_in_mini: int = 1
    grad_accum: ArrayTree | None = None

    ema_params: Any = None
    ema_decay: float = 0.9998
    use_pgd: bool = False

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





# @partial(jax.pmap, axis_name="batch")
# def validation_step(state: TrainState, batch: ArrayTree) -> ArrayTree:
#     metrics = state.apply_fn(
#         {"params": state.ema_params},
#         images=batch[0],
#         labels=jnp.where(batch[1] != -1, batch[1], 0),
#         det=True,
#     )
#     metrics["num_samples"] = batch[1] != -1
#     metrics = jax.tree_map(lambda x: (x * (batch[1] != -1)).sum(), metrics)
#     return jax.lax.psum(metrics, axis_name="batch")


# @partial(jax.pmap, axis_name="batch")
# def validation_adv_step(state: TrainState, batch: ArrayTree) -> ArrayTree:
#     rngs, updates = state.split_rngs()
#     metrics = state.apply_fn(
#         {"params": state.ema_params},
#         images=batch[0],
#         labels=jnp.where(batch[1] != -1, batch[1], 0),
#         det=True, use_pgd=False
#     )
#
#     metrics_adv = state.apply_fn(
#         {"params": state.ema_params},
#         images=batch[0],
#         labels=jnp.where(batch[1] != -1, batch[1], 0),
#         det=True, use_pgd=True, rngs=rngs,
#     )
#
#     metrics_adv = {'adv' + k: v for k, v in metrics_adv.items()}
#     metrics.update(metrics_adv)
#
#     metrics["num_samples"] = batch[1] != -1
#     metrics = jax.tree_util.tree_map(lambda x: (x * (batch[1] != -1)).sum(), metrics)
#     return jax.lax.psum(metrics, axis_name="batch")





def training_step(state: TrainState, batch: ArrayTree, use_pgd) -> tuple[TrainState, ArrayTree]:
    # jax.tree_util.tree_map(lambda x: jax.debug.inspect_array_sharding(x, callback=print), batch)
    # images,label=batch
    # print('images')
    # jax.debug.inspect_array_sharding(images, callback=print)
    # print('labels')
    # jax.debug.inspect_array_sharding(label, callback=print)

    def loss_fn(params: ArrayTree) -> ArrayTree:
        metrics = state.apply_fn({"params": params}, *batch, det=False, rngs=rngs, use_trade=not use_pgd,
                                 use_pgd=use_pgd, )
        metrics = jax.tree_map(jnp.mean, metrics)
        return metrics["loss"], metrics

    def update_fn(state: TrainState) -> TrainState:
        # Collect a global gradient from the accumulated gradients and apply actual
        # parameter update with resetting the accumulations to zero.
        grads = jax.tree_map(lambda g: g / state.micro_in_mini, state.grad_accum)
        state = state.apply_gradients(
            grads=grads,
            grad_accum=jax.tree_map(jnp.zeros_like, state.grad_accum),
            micro_step=state.micro_step % state.micro_in_mini,
        )
        new_ema_params = jax.tree_util.tree_map(
            lambda ema, normal: ema * state.ema_decay + (1 - state.ema_decay) * normal,
            state.ema_params, state.params)
        state = state.replace(ema_params=new_ema_params)

        return state

    rngs, updates = state.split_rngs()
    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    # metrics = jax.lax.pmean(metrics, axis_name="batch")

    # Update parameters with the gradients. If the gradient accumulation is enabled,
    # then the parameters will be updated at the end of each mini-batch step. In every
    # micro steps, the gradients will be accumulated.
    if state.grad_accum is None:
        state = state.apply_gradients(grads=grads)

        new_ema_params = jax.tree_util.tree_map(
            lambda ema, normal: ema * state.ema_decay + (1 - state.ema_decay) * normal,
            state.ema_params, state.params)
        state = state.replace(ema_params=new_ema_params)

    else:
        state = state.replace(
            grad_accum=jax.tree_map(lambda ga, g: ga + g, state.grad_accum, grads),
            micro_step=state.micro_step + 1,
        )
        state = jax.lax.cond(
            state.micro_step == state.micro_in_mini, update_fn, lambda x: x, state
        )
    return state.replace(**updates), metrics | state.opt_state.hyperparams




def training_step_kl(state: TrainState, batch: ArrayTree, use_pgd) -> tuple[TrainState, ArrayTree]:
    # jax.tree_util.tree_map(lambda x: jax.debug.inspect_array_sharding(x, callback=print), batch)
    # images,label=batch
    # print('images')
    # jax.debug.inspect_array_sharding(images, callback=print)
    # print('labels')
    # jax.debug.inspect_array_sharding(label, callback=print)

    def loss_fn(params: ArrayTree) -> ArrayTree:
        metrics = state.apply_fn({"params": params}, *batch, det=False, rngs=rngs, use_trade=not use_pgd,
                                 use_pgd=use_pgd,return_logits=True )


        metrics_ref = state.apply_fn({"params": state.ema_params}, *batch, det=False, rngs=rngs, use_trade=not use_pgd,
                                 use_pgd=use_pgd,return_logits=True ,ref=True)


        # print(metrics)
        # print(metrics_ref)

        kl_loss=optax.kl_divergence(flax.linen.log_softmax(metrics_ref.pop('logits'), axis=1), flax.linen.softmax(metrics.pop('logits'), axis=1))
        print(kl_loss.shape)

        metrics['kl_loss']=kl_loss
        metrics['ce_loss']=metrics['loss']
        metrics['loss']=metrics['ce_loss']+0.5*kl_loss
        metrics = jax.tree_map(jnp.mean, metrics)
        return metrics["loss"], metrics

    def update_fn(state: TrainState) -> TrainState:
        # Collect a global gradient from the accumulated gradients and apply actual
        # parameter update with resetting the accumulations to zero.
        grads = jax.tree_map(lambda g: g / state.micro_in_mini, state.grad_accum)
        state = state.apply_gradients(
            grads=grads,
            grad_accum=jax.tree_map(jnp.zeros_like, state.grad_accum),
            micro_step=state.micro_step % state.micro_in_mini,
        )
        new_ema_params = jax.tree_util.tree_map(
            lambda ema, normal: ema * state.ema_decay + (1 - state.ema_decay) * normal,
            state.ema_params, state.params)
        state = state.replace(ema_params=new_ema_params)

        return state

    rngs, updates = state.split_rngs()
    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    # metrics = jax.lax.pmean(metrics, axis_name="batch")

    # Update parameters with the gradients. If the gradient accumulation is enabled,
    # then the parameters will be updated at the end of each mini-batch step. In every
    # micro steps, the gradients will be accumulated.
    if state.grad_accum is None:
        state = state.apply_gradients(grads=grads)

        new_ema_params = jax.tree_util.tree_map(
            lambda ema, normal: ema * state.ema_decay + (1 - state.ema_decay) * normal,
            state.ema_params, state.params)
        state = state.replace(ema_params=new_ema_params)

    else:
        state = state.replace(
            grad_accum=jax.tree_map(lambda ga, g: ga + g, state.grad_accum, grads),
            micro_step=state.micro_step + 1,
        )
        state = jax.lax.cond(
            state.micro_step == state.micro_in_mini, update_fn, lambda x: x, state
        )
    return state.replace(**updates), metrics | state.opt_state.hyperparams




def training_step_test(state: TrainState, batch: ArrayTree, use_pgd) -> tuple[TrainState, ArrayTree]:
    # jax.tree_util.tree_map(lambda x: jax.debug.inspect_array_sharding(x, callback=print), batch)
    # images,label=batch
    # print('images')
    # jax.debug.inspect_array_sharding(images, callback=print)
    # print('labels')
    # jax.debug.inspect_array_sharding(label, callback=print)

    def loss_fn(params: ArrayTree) -> ArrayTree:
        metrics = state.apply_fn({"params": params}, *batch, det=False, rngs=rngs, use_trade=not use_pgd,
                                 use_pgd=use_pgd, )
        metrics = jax.tree_map(jnp.mean, metrics)
        return metrics["loss"], metrics

    def update_fn(state: TrainState) -> TrainState:
        # Collect a global gradient from the accumulated gradients and apply actual
        # parameter update with resetting the accumulations to zero.
        grads = jax.tree_map(lambda g: g / state.micro_in_mini, state.grad_accum)
        state = state.apply_gradients(
            grads=grads,
            grad_accum=jax.tree_map(jnp.zeros_like, state.grad_accum),
            micro_step=state.micro_step % state.micro_in_mini,
        )
        new_ema_params = jax.tree_util.tree_map(
            lambda ema, normal: ema * state.ema_decay + (1 - state.ema_decay) * normal,
            state.ema_params, state.params)
        state = state.replace(ema_params=new_ema_params)

        return state

    rngs, updates = state.split_rngs()
    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    # metrics = jax.lax.pmean(metrics, axis_name="batch")


    def rms(x):
        rms = jnp.sqrt(jnp.mean(jnp.square(x), )  +1e-7 )
        return rms


    g_embed=grads['model']['stem']['conv']['kernel']
    mu_embed=state.opt_state.inner_state[1][0].mu['model']['stem']['conv']['kernel']
    # mu_embed = state.opt_state.inner_state[0].mu['model']['stem']['conv']['kernel']
    metrics['g_embed']=rms(g_embed)
    metrics['mu_embed']=rms(mu_embed)
    metrics['g_d_mu_embed']=rms(g_embed/(mu_embed+1e-8))


    # Update parameters with the gradients. If the gradient accumulation is enabled,
    # then the parameters will be updated at the end of each mini-batch step. In every
    # micro steps, the gradients will be accumulated.
    if state.grad_accum is None:
        state = state.apply_gradients(grads=grads)

        new_ema_params = jax.tree_util.tree_map(
            lambda ema, normal: ema * state.ema_decay + (1 - state.ema_decay) * normal,
            state.ema_params, state.params)
        state = state.replace(ema_params=new_ema_params)

    else:
        state = state.replace(
            grad_accum=jax.tree_map(lambda ga, g: ga + g, state.grad_accum, grads),
            micro_step=state.micro_step + 1,
        )
        state = jax.lax.cond(
            state.micro_step == state.micro_in_mini, update_fn, lambda x: x, state
        )
    return state.replace(**updates), metrics | state.opt_state.hyperparams





def validation_adv_step(state: TrainState, batch: ArrayTree) -> ArrayTree:
    rngs, updates = state.split_rngs()
    labels=batch[1]
    batch[1]=jnp.where(batch[1] != -1, batch[1], 0)

    metrics = state.apply_fn(
        {"params": state.ema_params if state.ema_params is not None else state.params},
        # images=batch[0],
        # labels=jnp.where(batch[1] != -1, batch[1], 0),
        *batch,
        det=True, use_pgd=False
    )

    metrics_adv = state.apply_fn(
        {"params": state.ema_params if state.ema_params is not None else state.params},
        # images=batch[0],
        # labels=jnp.where(batch[1] != -1, batch[1], 0),
        *batch,
        det=True, use_pgd=True, rngs=rngs,
    )

    metrics_adv = {'adv' + k: v for k, v in metrics_adv.items()}
    metrics.update(metrics_adv)

    # metrics["num_samples"] = batch[1] != -1
    # metrics = jax.tree_util.tree_map(lambda x: (x * (batch[1] != -1)).sum(), metrics)

    metrics["num_samples"] = labels != -1
    metrics = jax.tree_util.tree_map(lambda x: (x * (labels != -1)).sum(), metrics)
    return metrics



def validation_step(state: TrainState, batch: ArrayTree) -> ArrayTree:
    metrics = state.apply_fn(
        {"params": state.ema_params},
        images=batch[0],
        labels=jnp.where(batch[1] != -1, batch[1], 0),
        det=True,
    )
    metrics["num_samples"] = batch[1] != -1
    metrics = jax.tree_map(lambda x: (x * (batch[1] != -1)).sum(), metrics)
    return metrics
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
from functools import partial
from typing import Callable

import einops
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from chex import Array, ArrayTree, PRNGKey
from flax.training import train_state
from flax.training.common_utils import shard_prng_key
from jax.tree_util import tree_map_with_path

from dataset import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from modeling import ViT
from utils import Mixup, get_layer_index_fn, load_pretrained_params, modified_lamb

from attacks.pgd import pgd_attack

CRITERION_COLLECTION = {
    "ce": optax.softmax_cross_entropy,
    "bce": lambda x, y: optax.sigmoid_binary_cross_entropy(x, y > 0).mean(-1),
}
OPTIMIZER_COLLECTION = {
    "adamw": optax.adamw,
    "lamb": modified_lamb,
}


class TrainState(train_state.TrainState):
    mixup_rng: PRNGKey
    dropout_rng: PRNGKey

    micro_step: int = 0
    micro_in_mini: int = 1
    grad_accum: ArrayTree | None = None

    def split_rngs(self) -> tuple[ArrayTree, ArrayTree]:
        mixup_rng, new_mixup_rng = jax.random.split(self.mixup_rng)
        dropout_rng, new_dropout_rng = jax.random.split(self.dropout_rng)

        rngs = {"mixup": mixup_rng, "dropout": dropout_rng}
        updates = {"mixup_rng": new_mixup_rng, "dropout_rng": new_dropout_rng}
        return rngs, updates

    def replicate(self) -> TrainState:
        return flax.jax_utils.replicate(self).replace(
            mixup_rng=shard_prng_key(self.mixup_rng),
            dropout_rng=shard_prng_key(self.dropout_rng),
        )


# class TrainModule(nn.Module):
#     model: ViT
#     mixup: Mixup
#     label_smoothing: float = 0.0
#     criterion: Callable[[Array, Array], Array] = CRITERION_COLLECTION["ce"]
#
#     def __call__(self, images: Array, labels: Array, det: bool = True) -> ArrayTree:
#         # Normalize the pixel values in TPU devices, instead of copying the normalized
#         # float values from CPU. This may reduce both memory usage and latency.
#         # images, labels = batch
#         images = jnp.moveaxis(images, 1, 3).astype(jnp.float32) / 0xFF
#
#         labels = nn.one_hot(labels, self.model.labels) if labels.ndim == 1 else labels
#         labels = labels.astype(jnp.float32)
#
#         if not det:
#             labels = optax.smooth_labels(labels, self.label_smoothing)
#             images, labels = self.mixup(images, labels)
#
#         logits = self.model(images, det=det)
#
#         return logits, labels

# def pgd_step(self, images: Array, det: bool = True) -> ArrayTree:
#     # Normalize the pixel values in TPU devices, instead of copying the normalized
#     # float values from CPU. This may reduce both memory usage and latency.
#     # images, labels = batch
#     images = jnp.moveaxis(images, 1, 3).astype(jnp.float32) / 0xFF
#     logits = self.model(images, det=det)
#
#     return logits


@partial(jax.pmap, axis_name="batch", donate_argnums=0)
def apply_model_trade(state, batch):
    key, updates = state.split_rngs()
    images, labels = batch
    images = einops.rearrange(images, 'b c h w->b h w c')
    images = images.astype(jnp.float32) / 255
    labels = labels.astype(jnp.float32)

    print(images.shape)

    """Computes gradients, loss and accuracy for a single batch."""
    adv_image = pgd_attack(images, labels, state, state.params, key=key, )

    # def loss_fn(params):
    #     logits = state.apply_fn({'params': params}, images)
    #     logits_adv = state.apply_fn({'params': params}, adv_image)
    #     one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    #     one_hot = optax.smooth_labels(one_hot, state.label_smoothing)
    #     loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    #     trade_loss = optax.kl_divergence(nn.log_softmax(logits_adv, axis=1), nn.softmax(logits, axis=1)).mean()
    #     metrics = {'loss': loss, 'trade_loss': trade_loss, 'logits': logits, 'logits_adv': logits_adv}
    #     return loss + state.trade_beta * trade_loss, metrics

    def loss_fn(params):
        logits_adv = state.apply_fn({'params': params}, adv_image)
        one_hot = jax.nn.one_hot(labels, logits.shape[-1])
        one_hot = optax.smooth_labels(one_hot, state.label_smoothing)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits_adv, labels=one_hot))
        metrics = {'loss': loss, }
        return loss, metrics

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    # accuracy_std = jnp.mean(jnp.argmax(metrics['logits'], -1) == labels)
    accuracy_adv = jnp.mean(jnp.argmax(metrics['logits_adv'], -1) == labels)

    # metrics['accuracy'] = accuracy_std
    metrics['adversarial accuracy'] = accuracy_adv

    metrics = jax.lax.pmean(metrics, axis_name="batch")

    grads = jax.lax.pmean(grads, axis_name="batch")

    state = state.apply_gradients(grads=grads)

    # new_ema_params = jax.tree_util.tree_map(
    #     lambda ema, normal: ema * state.ema_decay + (1 - state.ema_decay) * normal,
    #     state.ema_params, state.params)
    # state = state.replace(ema_params=new_ema_params)

    return state.replace(**updates), metrics | state.opt_state.hyperparams


# @partial(jax.pmap, axis_name="batch", donate_argnums=0)
# def training_step(state: TrainState, batch: ArrayTree) -> tuple[TrainState, ArrayTree]:
#     images, labels = batch
#     images = jnp.moveaxis(images, 1, 3).astype(jnp.float32) / 0xFF
#     images = pgd_attack(images, labels, state, state.params, )
#
#     def loss_fn(params: ArrayTree) -> ArrayTree:
#
#         logits, labels = state.apply_fn({"params": params}, images, batch[1], det=False, rngs=rngs)
#         loss = state.criterion(logits, labels)
#         labels = labels == labels.max(-1, keepdims=True)
#         preds = jax.lax.top_k(logits, k=5)[1]
#         accs = jnp.take_along_axis(labels, preds, axis=-1)
#         metrics = {"loss": loss, "acc1": accs[:, 0], "acc5": accs.any(-1)}
#         metrics = jax.tree_map(jnp.mean, metrics)
#         return metrics["loss"], metrics
#
#     def update_fn(state: TrainState) -> TrainState:
#         # Collect a global gradient from the accumulated gradients and apply actual
#         # parameter update with resetting the accumulations to zero.
#         grads = jax.tree_map(lambda g: g / state.micro_in_mini, state.grad_accum)
#         return state.apply_gradients(
#             grads=jax.lax.pmean(grads, axis_name="batch"),
#             grad_accum=jax.tree_map(jnp.zeros_like, state.grad_accum),
#             micro_step=state.micro_step % state.micro_in_mini,
#         )
#
#     rngs, updates = state.split_rngs()
#     (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
#     metrics = jax.lax.pmean(metrics, axis_name="batch")
#
#     # Update parameters with the gradients. If the gradient accumulation is enabled,
#     # then the parameters will be updated at the end of each mini-batch step. In every
#     # micro steps, the gradients will be accumulated.
#     if state.grad_accum is None:
#         state = state.apply_gradients(grads=jax.lax.pmean(grads, axis_name="batch"))
#     else:
#         state = state.replace(
#             grad_accum=jax.tree_map(lambda ga, g: ga + g, state.grad_accum, grads),
#             micro_step=state.micro_step + 1,
#         )
#         state = jax.lax.cond(
#             state.micro_step == state.micro_in_mini, update_fn, lambda x: x, state
#         )
#     return state.replace(**updates), metrics | state.opt_state.hyperparams


# @partial(jax.pmap, axis_name="batch")
# def validation_step(state: TrainState, batch: ArrayTree) -> ArrayTree:
#     images, labels = batch
#     images = einops.rearrange(images, 'b c h w->b h w c')
#     images = images.astype(jnp.float32) / 255
#     labels = labels.astype(jnp.float32)
#
#     metrics_adv = state.apply_fn(
#         {"params": state.params},
#         images=pgd_attack(images, batch[1], state, state.params, step_size=1 / 255, maxiter=10),
#         labels=jnp.where(batch[1] != -1, batch[1], 0),
#         det=True
#     )
#
#     metrics = state.apply_fn(
#         {"params": state.params},
#         images=images,
#         labels=jnp.where(batch[1] != -1, batch[1], 0),
#         det=True, use_pgd=True
#     )
#
#     metrics['val_adv_acc1'] = metrics_adv['acc1']
#
#     metrics["num_samples"] = batch[1] != -1
#     metrics = jax.tree_map(lambda x: (x * (batch[1] != -1)).sum(), metrics)
#     return jax.lax.psum(metrics, axis_name="batch")


@partial(jax.pmap, axis_name="batch", )
def validation_step(state, data):
    # inputs, labels = data

    inputs, labels = data
    inputs = inputs.astype(jnp.float32)
    labels = labels.astype(jnp.int64)

    # print(images)
    # while True:
    #     pass
    inputs = einops.rearrange(inputs, 'b c h w->b h w c')

    logits = state.apply_fn({"params": state.ema_params}, inputs)
    clean_accuracy = jnp.argmax(logits, axis=-1) == labels

    maxiter = 10
    EPSILON = 4 / 255

    adversarial_images = pgd_attack(inputs, labels, state, epsilon=EPSILON, maxiter=maxiter,
                                    step_size=EPSILON * 2 / maxiter)
    logits_adv = state.apply_fn({"params": state.ema_params}, adversarial_images)
    adversarial_accuracy = jnp.argmax(logits_adv, axis=-1) == labels

    metrics = {"adversarial accuracy": adversarial_accuracy, "accuracy": clean_accuracy, "num_samples": labels != -1}

    metrics = jax.tree_util.tree_map(lambda x: (x * (labels != -1)).sum(), metrics)

    metrics = jax.lax.psum(metrics, axis_name='batch')

    # metrics = jax.lax.pmean(metrics, axis_name='batch')
    return metrics


def create_train_state(args: argparse.Namespace) -> TrainState:
    model = ViT(
        layers=args.layers,
        dim=args.dim,
        heads=args.heads,
        labels=args.labels,
        layerscale=args.layerscale,
        patch_size=args.patch_size,
        image_size=args.image_size,
        posemb=args.posemb,
        pooling=args.pooling,
        dropout=args.dropout,
        droppath=args.droppath,
        grad_ckpt=args.grad_ckpt,
    )
    module = TrainModule(
        model=model,
        mixup=Mixup(args.mixup, args.cutmix),
        label_smoothing=args.label_smoothing if args.criterion == "ce" else 0,
        criterion=CRITERION_COLLECTION[args.criterion],
    )

    # Initialize the model weights with dummy inputs. Using the init RNGS and inputs, we
    # will tabulate the summary of model and its parameters. Furthermore, empty gradient
    # accumulation arrays will be prepared if the gradient accumulation is enabled.
    example_inputs = {
        "images": jnp.zeros((1, 3, args.image_size, args.image_size,), dtype=jnp.int8),
        "labels": jnp.zeros((1,), dtype=jnp.int32),
    }
    init_rngs = {"params": jax.random.PRNGKey(args.init_seed)}
    # print(module.tabulate(init_rngs, **example_inputs))

    params = module.init(init_rngs, **example_inputs)["params"]
    if args.pretrained_ckpt is not None:
        params = load_pretrained_params(args, params)
    if args.grad_accum > 1:
        grad_accum = jax.tree_map(jnp.zeros_like, params)

    # Create learning rate scheduler and optimizer with gradient clipping. The learning
    # rate will be recorded at `hyperparams` by `optax.inject_hyperparameters`.
    @partial(optax.inject_hyperparams, hyperparam_dtype=jnp.float32)
    def create_optimizer_fn(
            learning_rate: optax.Schedule,
    ) -> optax.GradientTransformation:
        tx = OPTIMIZER_COLLECTION[args.optimizer](
            learning_rate=learning_rate,
            b1=args.adam_b1,
            b2=args.adam_b2,
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
            mask=partial(tree_map_with_path, lambda kp, *_: kp[-1].key == "kernel"),
        )
        if args.lr_decay < 1.0:
            layerwise_scales = {
                i: optax.scale(args.lr_decay ** (args.layers - i))
                for i in range(args.layers + 1)
            }
            label_fn = partial(get_layer_index_fn, num_layers=args.layers)
            label_fn = partial(tree_map_with_path, label_fn)
            tx = optax.chain(tx, optax.multi_transform(layerwise_scales, label_fn))
        if args.clip_grad > 0:
            tx = optax.chain(optax.clip_by_global_norm(args.clip_grad), tx)
        return tx

    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=args.learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.training_steps,
        end_value=1e-5,
    )
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=create_optimizer_fn(learning_rate),
        mixup_rng=jax.random.PRNGKey(args.mixup_seed + jax.process_index()),
        dropout_rng=jax.random.PRNGKey(args.dropout_seed + jax.process_index()),
        micro_step=0,
        micro_in_mini=args.grad_accum,
        grad_accum=grad_accum if args.grad_accum > 1 else None,
    )

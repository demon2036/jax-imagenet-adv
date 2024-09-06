from __future__ import annotations

from typing import Callable, Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from chex import Array, ArrayTree

from attacks import pgd_attack, trade
from utils import Mixup

CRITERION_COLLECTION = {
    "ce": optax.softmax_cross_entropy,
    "bce": lambda x, y: optax.sigmoid_binary_cross_entropy(x, y > 0).mean(-1),
}


class TrainModule(nn.Module):
    model: Any
    mixup: Mixup
    label_smoothing: float = 0.0
    criterion: Callable[[Array, Array], Array] = CRITERION_COLLECTION["ce"]

    def __call__(self, images: Array, labels: Array, det: bool = True) -> ArrayTree:
        # Normalize the pixel values in TPU devices, instead of copying the normalized
        # float values from CPU. This may reduce both memory usage and latency.
        images = jnp.moveaxis(images, 1, 3).astype(jnp.float32) / 0xFF

        labels = nn.one_hot(labels, self.model.labels) if labels.ndim == 1 else labels
        labels = labels.astype(jnp.float32)

        if not det:
            labels = optax.smooth_labels(labels, self.label_smoothing)
            images, labels = self.mixup(images, labels)

        loss = self.criterion((logits := self.model(images, det=det)), labels)
        labels = labels == labels.max(-1, keepdims=True)

        # Instead of directly comparing the maximum classes of predicted logits with the
        # given one-hot labels, we will check if the predicted classes are within the
        # label set. This approach is equivalent to traditional methods in single-label
        # classification and also supports multi-label tasks.
        preds = jax.lax.top_k(logits, k=5)[1]
        accs = jnp.take_along_axis(labels, preds, axis=-1)
        return {"loss": loss, "acc1": accs[:, 0], "acc5": accs.any(-1)}


# class TrainAdvModule(nn.Module):
#     model: Any
#     mixup: Mixup
#     label_smoothing: float = 0.0
#     criterion: Callable[[Array, Array], Array] = CRITERION_COLLECTION["ce"]
#
#     def __call__(self, images: Array, labels: Array, det: bool = True, use_pgd=True) -> ArrayTree:
#         # Normalize the pixel values in TPU devices, instead of copying the normalized
#         # float values from CPU. This may reduce both memory usage and latency.
#         images = jnp.moveaxis(images, 1, 3).astype(jnp.float32) / 0xFF
#
#         labels = nn.one_hot(labels, self.model.labels) if labels.ndim == 1 else labels
#         labels = labels.astype(jnp.float32)
#
#         if not det:
#             labels = optax.smooth_labels(labels, self.label_smoothing)
#             images, labels = self.mixup(images, labels)
#
#         if use_pgd:
#             images = pgd_attack(images, labels, self.model, key=self.make_rng('adv'))
#
#         loss = self.criterion((logits := self.model(images, det=det)), labels)
#         labels = labels == labels.max(-1, keepdims=True)
#
#         # Instead of directly comparing the maximum classes of predicted logits with the
#         # given one-hot labels, we will check if the predicted classes are within the
#         # label set. This approach is equivalent to traditional methods in single-label
#         # classification and also supports multi-label tasks.
#         preds = jax.lax.top_k(logits, k=5)[1]
#         accs = jnp.take_along_axis(labels, preds, axis=-1)
#         return {"loss": loss, "acc1": accs[:, 0], "acc5": accs.any(-1)}



class TrainAdvModule(nn.Module):
    model: Any
    mixup: Mixup
    label_smoothing: float = 0.0
    criterion: Callable[[Array, Array], Array] = CRITERION_COLLECTION["ce"]


    # train_adv_step:int=3
    # train_adv_step_size: float = 4 / 3 / 255

    train_adv_step: int = 10
    train_adv_step_size: float = 1 / 255

    # test_adv_step: int = 10
    #
    # test_adv_step: int = 10

    # test_adv_step_size: float = 1 / 255

    eps: float = 4 / 255


    def __call__(self, images: Array, labels: Array, det: bool = True, use_pgd=True,use_trade=False,train=False) -> ArrayTree:
        # Normalize the pixel values in TPU devices, instead of copying the normalized
        # float values from CPU. This may reduce both memory usage and latency.
        images = jnp.moveaxis(images, 1, 3).astype(jnp.float32) / 0xFF

        labels = nn.one_hot(labels, self.model.labels) if labels.ndim == 1 else labels
        labels = labels.astype(jnp.float32)

        if not det:
            labels = optax.smooth_labels(labels, self.label_smoothing)
            images, labels = self.mixup(images, labels)


        if use_trade:
            x_adv = trade(images, self.model,  key=self.make_rng('adv'),
                            step_size = self.train_adv_step_size,  # if train else self.test_adv_step_size ,
                            maxiter = self.train_adv_step  # if train else self.test_adv_step
                          )
            logits =self.model(images)
            logits_adv=self.model(x_adv)
            loss_ce= jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels))
            trade_loss = optax.kl_divergence(nn.log_softmax(logits_adv, axis=1), nn.softmax(logits, axis=1)).mean()
            labels = labels == labels.max(-1, keepdims=True)

            preds = jax.lax.top_k(logits, k=5)[1]
            accs = jnp.take_along_axis(labels, preds, axis=-1)
            return {"loss": loss_ce+8*trade_loss, "loss_ce":loss_ce,"trade_loss":trade_loss,    "acc1": accs[:, 0], "acc5": accs.any(-1)}
        else:

            if use_pgd:
                images = pgd_attack(images, labels, self.model, key=self.make_rng('adv'),
                                    step_size=self.train_adv_step_size, #if train else self.test_adv_step_size ,
                                    maxiter=self.train_adv_step #if train else self.test_adv_step
                )

            loss = self.criterion((logits := self.model(images, det=det)), labels)
            labels = labels == labels.max(-1, keepdims=True)

            # Instead of directly comparing the maximum classes of predicted logits with the
            # given one-hot labels, we will check if the predicted classes are within the
            # label set. This approach is equivalent to traditional methods in single-label
            # classification and also supports multi-label tasks.
            preds = jax.lax.top_k(logits, k=5)[1]
            accs = jnp.take_along_axis(labels, preds, axis=-1)
            return {"loss": loss, "acc1": accs[:, 0], "acc5": accs.any(-1)}
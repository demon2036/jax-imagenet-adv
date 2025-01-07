from __future__ import annotations

from typing import Callable, Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from chex import Array, ArrayTree

from attacks import pgd_attack, trade
from attacks.pgd import pgd_dynamic_scale_attack
from attacks.trades import trade_lse
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

    def __call__(self, images: Array, labels: Array, det: bool = True,*args,**kwargs) -> ArrayTree:
        # Normalize the pixel values in TPU devices, instead of copying the normalized
        # float values from CPU. This may reduce both memory usage and latency.

        # if isinstance(images,jax._src.interpreters.partial_eval.DynamicJaxprTracer):
        #     # jax.debug.visualize_array_sharding(x[0])
        #     print(images.shape)
        #     jax.debug.inspect_array_sharding(images,callback=print)
        # else:
        #     print(type(images))


        images = jnp.moveaxis(images, 1, 3).astype(jnp.float32) / 0xFF

        labels = nn.one_hot(labels, self.model.labels) if labels.ndim == 1 else labels
        labels = labels.astype(jnp.float32)

        if not det:
            labels = optax.smooth_labels(labels, self.label_smoothing)
            images, labels = self.mixup(images, labels)


        # out=self.model(images, det=det)
        #
        # loss = out-jnp.ones_like(out)
        # return {"loss": loss, }


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
    use_pgd: bool = False
    beta: float = 0.0

    def __call__(self, images: Array, labels: Array, det: bool = True, use_pgd=True, use_trade=False,
                 ref=False,return_logits=False) -> ArrayTree:
        # Normalize the pixel values in TPU devices, instead of copying the normalized
        # float values from CPU. This may reduce both memory usage and latency.
        images = jnp.moveaxis(images, 1, 3).astype(jnp.float32) / 0xFF

        labels = nn.one_hot(labels, self.model.labels) if labels.ndim == 1 else labels
        labels = labels.astype(jnp.float32)

        if not det:
            labels = optax.smooth_labels(labels, self.label_smoothing)
            images, labels = self.mixup(images, labels)
        print(use_pgd,use_trade,)



        if ref:
            pass
        else:
            if use_trade:
                logits_natural = nn.softmax(self.model(images),axis=1)
                x_adv = trade_lse(images, self.model, key=self.make_rng('adv'),
                                  step_size=self.train_adv_step_size,  # if train else self.test_adv_step_size ,
                                  maxiter=self.train_adv_step,logits=jax.lax.stop_gradient(logits_natural)  # if train else self.test_adv_step
                                  )

                logits_adv = nn.softmax(self.model(x_adv),axis=1)

                loss_natural = jnp.sum((logits_natural - labels) ** 2, axis=-1)
                loss_robust = jnp.sum((logits_adv - logits_natural) ** 2, axis=-1)
                loss_robust = nn.relu(loss_robust - 0)
                loss = loss_natural.mean() + self.beta * loss_robust.mean()

                labels = labels == labels.max(-1, keepdims=True)
                #
                preds = jax.lax.top_k(logits_adv, k=5)[1]
                accs = jnp.take_along_axis(labels, preds, axis=-1)
                return {"loss": loss, "acc1": accs[:, 0], "acc5": accs.any(-1),'loss_natural':loss_natural.mean(),'loss_robust':loss_robust.mean()}


            elif use_trade:
                x_adv = trade(images, self.model, key=self.make_rng('adv'),
                              step_size=self.train_adv_step_size,  # if train else self.test_adv_step_size ,
                              maxiter=self.train_adv_step  # if train else self.test_adv_step
                              )
                logits = self.model(images)
                logits_adv = self.model(x_adv)
                loss_ce = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels))
                trade_loss = optax.kl_divergence(nn.log_softmax(logits_adv, axis=1), nn.softmax(logits, axis=1)).mean()
                labels = labels == labels.max(-1, keepdims=True)

                preds = jax.lax.top_k(logits, k=5)[1]
                accs = jnp.take_along_axis(labels, preds, axis=-1)
                return {"loss": loss_ce + 5 * trade_loss, "loss_ce": loss_ce, "trade_loss": trade_loss, "acc1": accs[:, 0],
                        "acc5": accs.any(-1)}
            else:

                if use_pgd:
                    # images = pgd_attack(images, labels, self.model, key=self.make_rng('adv'),epsilon=self.eps,
                    #                     step_size=self.train_adv_step_size,  #if train else self.test_adv_step_size ,
                    #                     maxiter=self.train_adv_step  #if train else self.test_adv_step
                    #                     )

                    def test1():
                        return pgd_attack(images, labels, self.model, key=self.make_rng('adv'), epsilon=self.eps,
                                          step_size=self.train_adv_step_size,  # if train else self.test_adv_step_size ,
                                          maxiter=self.train_adv_step   # if train else self.test_adv_step
                                          )
                    def test2():
                        return  pgd_attack(images, labels, self.model, key=self.make_rng('adv'),epsilon=self.eps,
                                        step_size=12/8/255,  #if train else self.test_adv_step_size ,
                                        maxiter=8  #if train else self.test_adv_step
                                        )


                    # images=jax.lax.cond(jax.random.uniform(self.make_rng('adv'),(1,))[0]<0.9,test1,test2   )

                    images = pgd_attack(images, labels, self.model, key=self.make_rng('adv'),epsilon=self.eps,
                                        step_size=self.train_adv_step_size,  #if train else self.test_adv_step_size ,
                                        maxiter=self.train_adv_step  #if train else self.test_adv_step
                                        )


        loss = self.criterion((logits := self.model(images, det=det)), labels)
        labels = labels == labels.max(-1, keepdims=True)

        # Instead of directly comparing the maximum classes of predicted logits with the
        # given one-hot labels, we will check if the predicted classes are within the
        # label set. This approach is equivalent to traditional methods in single-label
        # classification and also supports multi-label tasks.
        preds = jax.lax.top_k(logits, k=5)[1]
        accs = jnp.take_along_axis(labels, preds, axis=-1)
        if return_logits:
            return {"loss": loss, "acc1": accs[:, 0], "acc5": accs.any(-1),'logits':logits}
        else:
            return {"loss": loss, "acc1": accs[:, 0], "acc5": accs.any(-1)}















class TrainAdvModule2(nn.Module):

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
    use_pgd: bool = False
    beta: float = 0.0

    def __call__(self, images: Array, labels: Array, det: bool = True, use_pgd=True, use_trade=False,
                 ref=False,return_logits=False) -> ArrayTree:
        # Normalize the pixel values in TPU devices, instead of copying the normalized
        # float values from CPU. This may reduce both memory usage and latency.
        images = jnp.moveaxis(images, 1, 3).astype(jnp.float32) / 0xFF

        labels = nn.one_hot(labels, self.model.labels) if labels.ndim == 1 else labels
        labels = labels.astype(jnp.float32)

        if not det:
            labels = optax.smooth_labels(labels, self.label_smoothing)
            images, labels = self.mixup(images, labels)
        print(use_pgd,use_trade,)



        if ref:
            pass
        else:
            images = pgd_dynamic_scale_attack(images, labels, self.model, key=self.make_rng('adv'),epsilon=self.eps,
                                step_size=self.train_adv_step_size,  #if train else self.test_adv_step_size ,
                                maxiter=self.train_adv_step,  #if train else self.test_adv_step
                                              dynamic=not det
                                )


        loss = self.criterion((logits := self.model(images, det=det)), labels)
        labels = labels == labels.max(-1, keepdims=True)

        # Instead of directly comparing the maximum classes of predicted logits with the
        # given one-hot labels, we will check if the predicted classes are within the
        # label set. This approach is equivalent to traditional methods in single-label
        # classification and also supports multi-label tasks.
        preds = jax.lax.top_k(logits, k=5)[1]
        accs = jnp.take_along_axis(labels, preds, axis=-1)
        if return_logits:
            return {"loss": loss, "acc1": accs[:, 0], "acc5": accs.any(-1),'logits':logits}
        else:
            return {"loss": loss, "acc1": accs[:, 0], "acc5": accs.any(-1)}

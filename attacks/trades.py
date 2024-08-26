import os
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn



def loss_fun_trade(model, data):
    """Compute the loss of the network."""
    inputs, logits = data
    x_adv = inputs.astype(jnp.float32)
    logits_adv = model(x_adv)
    return optax.kl_divergence(nn.log_softmax(logits_adv, axis=1), nn.softmax(logits, axis=1)).mean()


def trade(image, model, epsilon=4/255, maxiter=10, step_size=1/255, key=None):
    logits = jax.lax.stop_gradient(model(image))

    # x_adv = 0.001 * jax.random.normal(key, shape=image.shape) + image

    x_adv = jax.random.uniform(key, shape=image.shape, minval=-epsilon, maxval=epsilon) + image
    x_adv = jnp.clip(x_adv, 0, 1)

    # def adversarial_loss(adv_image, image):
    #     return loss_fun_trade(state, (image, adv_image, label))

    def adversarial_loss(adv_image, logits):
        return loss_fun_trade(model, (adv_image, logits))

    grad_adversarial = jax.grad(adversarial_loss)
    for _ in range(maxiter):
        # compute gradient of the loss wrt to the image
        sign_grad = jnp.sign(jax.lax.stop_gradient(grad_adversarial(x_adv, logits)))
        # heuristic step-size 2 eps / maxiter
        # image_perturbation += step_size * sign_grad

        # delta = jnp.clip(image_perturbation - image, min=-epsilon, max=epsilon)

        x_adv = jax.lax.stop_gradient(x_adv) + step_size * sign_grad
        r1 = jnp.where(x_adv > image - epsilon, x_adv, image - epsilon)
        x_adv = jnp.where(r1 < image + epsilon, r1, image + epsilon)

        x_adv = jnp.clip(x_adv, min=0, max=1)

        # projection step onto the L-infinity ball centered at image
        # image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)

    # clip the image to ensure pixels are between 0 and 1
    return jax.lax.stop_gradient(x_adv)




# @partial(jax.pmap, axis_name="batch")
# def apply_model_trade(state, data, key):
#     images, labels = data
#
#     images = einops.rearrange(images, 'b c h w->b h w c')
#
#     images = images.astype(jnp.float32) / 255
#     labels = labels.astype(jnp.float32)
#
#     print(images.shape)
#
#     """Computes gradients, loss and accuracy for a single batch."""
#     adv_image = trade(images, labels, state, key=key, epsilon=EPSILON, step_size=2 / 255)
#
#     def loss_fn(params):
#         logits = state.apply_fn({'params': params}, images)
#         logits_adv = state.apply_fn({'params': params}, adv_image)
#         one_hot = jax.nn.one_hot(labels, logits.shape[-1])
#         one_hot = optax.smooth_labels(one_hot, state.label_smoothing)
#         loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
#         trade_loss = optax.kl_divergence(nn.log_softmax(logits_adv, axis=1), nn.softmax(logits, axis=1)).mean()
#         metrics = {'loss': loss, 'trade_loss': trade_loss, 'logits': logits, 'logits_adv': logits_adv}
#
#         return loss + state.trade_beta * trade_loss, metrics



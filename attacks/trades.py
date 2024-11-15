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


def trade(image, model, epsilon=4/255, maxiter=3, step_size=4/3/255, key=None):
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




def trade_lse(image, model, epsilon=4/255, maxiter=3, step_size=4/3/255, key=None):
    logits = jax.lax.stop_gradient(model(image))

    # x_adv = 0.001 * jax.random.normal(key, shape=image.shape) + image

    x_adv = jax.random.uniform(key, shape=image.shape, minval=-epsilon, maxval=epsilon) + image
    x_adv = jnp.clip(x_adv, 0, 1)

    # def adversarial_loss(adv_image, image):
    #     return loss_fun_trade(state, (image, adv_image, label))

    def adversarial_loss(adv_image, logits):
        logits_adv = model(adv_image)
        return jnp.sum((logits_adv-logits)**2)


    grad_adversarial = jax.grad(adversarial_loss)
    for _ in range(maxiter):
        # compute gradient of the loss wrt to the image
        sign_grad = jnp.sign(jax.lax.stop_gradient(grad_adversarial(x_adv, logits)))
        # heuristic step-size 2 eps / maxiter

        x_adv = jax.lax.stop_gradient(x_adv) + step_size * sign_grad
        r1 = jnp.where(x_adv > image - epsilon, x_adv, image - epsilon)
        x_adv = jnp.where(r1 < image + epsilon, r1, image + epsilon)

        x_adv = jnp.clip(x_adv, min=0, max=1)
        # projection step onto the L-infinity ball centered at image
        # image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)

    # clip the image to ensure pixels are between 0 and 1
    return jax.lax.stop_gradient(x_adv)



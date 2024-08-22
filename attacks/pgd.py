import einops
import jax.numpy as jnp
from optax.losses import softmax_cross_entropy_with_integer_labels
import jax
import optax


def pgd_attack(image, label, model, epsilon=4 / 255, step_size=4 / 255, maxiter=1, key=None):
    """PGD attack on the L-infinity ball with radius epsilon.

  Args:
    image: array-like, input data for the CNN
    label: integer, class label corresponding to image
    params: tree, parameters of the model to attack
    epsilon: float, radius of the L-infinity ball.
    maxiter: int, number of iterations of this algorithm.

  Returns:
    perturbed_image: Adversarial image on the boundary of the L-infinity ball
      of radius epsilon and centered at image.

  Notes:
    PGD attack is described in (Madry et al. 2017),
    https://arxiv.org/pdf/1706.06083.pdf
    :param state:
    :param image:
    :param label:
    :param params:
    :param maxiter:
    :param epsilon:
    :param step_size:
  """

    # image = einops.rearrange(image, 'b c h w->b h w c')
    # image = image.astype(jnp.float32)
    # label = label.astype(jnp.int32)

    # image_perturbation = jnp.zeros_like(image)
    image_perturbation = jax.random.uniform(key, image.shape, minval=-epsilon, maxval=epsilon)

    def adversarial_loss(perturbation):
        logits = model(image + perturbation)
        # print(logits.shape,label.shape)
        loss_value = jnp.mean(optax.softmax_cross_entropy(logits, label))
        # loss_value = logits
        return loss_value

    # for _ in range(maxiter):
    #     # compute gradient of the loss wrt to the image
    #     sign_grad = jnp.sign(adversarial_loss(image_perturbation))

    grad_adversarial = jax.grad(adversarial_loss)
    for _ in range(maxiter):
        # compute gradient of the loss wrt to the image
        sign_grad = jnp.sign(grad_adversarial(image_perturbation))

        # heuristic step-size 2 eps / maxiter
        image_perturbation += step_size * sign_grad
        # projection step onto the L-infinity ball centered at image
        image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)

    # sign_grad = jnp.sign(grad_adversarial(image_perturbation))
    # image_perturbation += step_size * sign_grad
    # image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)

    # clip the image to ensure pixels are between 0 and 1
    image_perturbation = jnp.clip(image + image_perturbation, 0, 1)
    return jax.lax.stop_gradient(image_perturbation)

#image, label, state, epsilon=8 / 255, step_size=2 / 255, maxiter=10
# def pgd_attack_l2(image, label, state, epsilon=128 / 255, maxiter=10,  key=None):
#     delta = 0.001 * jax.random.normal(key, image.shape)
#     optimizer = optax.sgd(epsilon / maxiter * 2)
#     opt_state = optimizer.init(delta)
#     # p_natural = state.apply_fn({'params': state.ema_params}, x)
#     def jax_re_norm(delta, max_norm):
#         b, h, w, c = delta.shape
#         norms = jnp.linalg.norm(delta.reshape(b, -1), ord=2, axis=1, keepdims=True).reshape(b, 1, 1, 1)
#         desired = jnp.clip(norms, a_min=None, a_max=max_norm)
#         scale = desired / (1e-6 + norms)
#         return delta * scale
#
#     def grad_fn(delta, x):
#         adv = x + delta
#         model_out = state.apply_fn({'params': state.ema_params}, adv)
#         # loss = -1 * optax.losses.kl_divergence(nn.log_softmax(model_out), p_natural)
#         loss_value = -jnp.mean(softmax_cross_entropy_with_integer_labels(model_out, label))
#         return loss_value
#
#     for _ in range(maxiter):
#         grad = jax.grad(grad_fn)(delta, image)
#         grad_norm = jnp.linalg.norm(grad.reshape(grad.shape[0], -1), axis=1)
#         grad = grad / grad_norm.reshape(-1, 1, 1, 1)
#         updates, opt_state = optimizer.update(grad, opt_state, delta)
#         delta = optax.apply_updates(delta, updates)
#         delta = delta + image
#         delta = jnp.clip(delta, 0, 1) - image
#         delta = jax_re_norm(delta, max_norm=epsilon)
#
#     return jnp.clip(image + delta, 0, 1)

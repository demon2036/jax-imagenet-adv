import einops
import jax.numpy as jnp
import numpy as np
from optax.losses import softmax_cross_entropy_with_integer_labels
import jax
import optax


# def pgd_attack(image, label, model, epsilon=4 / 255, step_size=4/3 / 255, maxiter=3, key=None):
#     """PGD attack on the L-infinity ball with radius epsilon.
#
#   Args:
#     image: array-like, input data for the CNN
#     label: integer, class label corresponding to image
#     params: tree, parameters of the model to attack
#     epsilon: float, radius of the L-infinity ball.
#     maxiter: int, number of iterations of this algorithm.
#
#   Returns:
#     perturbed_image: Adversarial image on the boundary of the L-infinity ball
#       of radius epsilon and centered at image.
#
#   Notes:
#     PGD attack is described in (Madry et al. 2017),
#     https://arxiv.org/pdf/1706.06083.pdf
#     :param state:
#     :param image:
#     :param label:
#     :param params:
#     :param maxiter:
#     :param epsilon:
#     :param step_size:
#   """
#
#     # image = einops.rearrange(image, 'b c h w->b h w c')
#     # image = image.astype(jnp.float32)
#     # label = label.astype(jnp.int32)
#
#     # image_perturbation = jnp.zeros_like(image)
#     image_perturbation = jax.random.uniform(key, image.shape, minval=-epsilon, maxval=epsilon)
#
#     # print(label)
#
#     def adversarial_loss(perturbation):
#         logits = model(jnp.clip(image + perturbation, 0, 1))
#         # print(logits.shape,label.shape)
#         loss_value = jnp.mean(optax.softmax_cross_entropy(logits, label))
#         # loss_value = logits
#         return loss_value
#
#     # for _ in range(maxiter):
#     #     # compute gradient of the loss wrt to the image
#     #     sign_grad = jnp.sign(adversarial_loss(image_perturbation))
#
#     grad_adversarial = jax.grad(adversarial_loss)
#     for _ in range(maxiter):
#         # compute gradient of the loss wrt to the image
#         sign_grad = jnp.sign(grad_adversarial(image_perturbation))
#
#         # heuristic step-size 2 eps / maxiter
#         image_perturbation += step_size * sign_grad
#         # projection step onto the L-infinity ball centered at image
#         image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)
#
#     # sign_grad = jnp.sign(grad_adversarial(image_perturbation))
#     # image_perturbation += step_size * sign_grad
#     # image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)
#
#     # clip the image to ensure pixels are between 0 and 1
#     image_perturbation = jnp.clip(image + image_perturbation, 0, 1)
#     return jax.lax.stop_gradient(image_perturbation)






# def _nucleus_sampling(p: float=0.6, t: float = 1.0,beta=1/4, *, logits):
#   logits = logits / t
#
#   logits=jnp.abs(logits)
#
#   neg_inf = np.array(-1.0e7)  # Effective negative infinity.
#   logits_sorted = jnp.sort(logits, axis=-1, descending=True)
#   sorted_cum_probs = jnp.cumsum(
#       jax.nn.softmax(logits_sorted, axis=-1), axis=-1)
#   cutoff_index = jnp.sum(sorted_cum_probs < p, axis=-1, keepdims=True)
#   cutoff_logit = jnp.take_along_axis(logits_sorted, cutoff_index, axis=-1)
#   # logits = jnp.where(logits < cutoff_logit,
#   #                    jnp.full_like(logits, neg_inf), logits)
#   return (logits < cutoff_logit).mean(),jnp.where(logits < cutoff_logit,
#                      jnp.full_like(logits, beta), 1.0)





# def pgd_dynamic_scale_attack(image, label, model, epsilon=4 / 255, step_size=4/3 / 255, maxiter=3, key=None,dynamic=False):
#     """PGD attack on the L-infinity ball with radius epsilon.
#
#   Args:
#     image: array-like, input data for the CNN
#     label: integer, class label corresponding to image
#     params: tree, parameters of the model to attack
#     epsilon: float, radius of the L-infinity ball.
#     maxiter: int, number of iterations of this algorithm.
#
#   Returns:
#     perturbed_image: Adversarial image on the boundary of the L-infinity ball
#       of radius epsilon and centered at image.
#
#   Notes:
#     PGD attack is described in (Madry et al. 2017),
#     https://arxiv.org/pdf/1706.06083.pdf
#     :param state:
#     :param image:
#     :param label:
#     :param params:
#     :param maxiter:
#     :param epsilon:
#     :param step_size:
#   """
#     b,h,w,c=image.shape
#
#     # image = einops.rearrange(image, 'b c h w->b h w c')
#     # image = image.astype(jnp.float32)
#     # label = label.astype(jnp.int32)
#
#     # image_perturbation = jnp.zeros_like(image)
#
#     key1,key2,key3,key4,key5=jax.random.split(key,5)
#
#     image_perturbation = jax.random.uniform(key1, image.shape, minval=-epsilon, maxval=epsilon)
#
#
#     if dynamic:
#     #     # image_perturbation_zero=jnp.zeros_like(image)
#     #     # image_perturbation = jnp.concatenate([image_perturbation[None,...], image_perturbation_zero[None,...]], axis=0)
#     #     # image_perturbation=jax.random.choice(key2, image_perturbation, p=jnp.array([0.5, 0.5]))
#     #
#     #
#     #     # step_size=jax.random.uniform(key2,(1,),minval=0.5,maxval=1).reshape((-1,1,1,1))*step_size
#         adv_step_size=jax.random.uniform(key2,(1,),minval=0.75,maxval=1).reshape((-1,1,1,1))*step_size
#         # epsilon = jax.random.uniform(key4, (1,), minval = 1.0, maxval = 1.5).reshape((-1,1,1,1)) *epsilon
#         # epsilon_extend = jax.random.uniform(key4, (image.shape[0],), minval=1.5, maxval=2.0) * epsilon
#
#         # r = jax.random.uniform(key5, shape=(image.shape[0],))
#         # sorted_indices = jnp.argsort(r)
#         # mask = jnp.zeros_like(sorted_indices, dtype=bool)
#         # mask = mask.at[sorted_indices[:int(image.shape[0]*0.1)]].set(True)
#         # epsilon = jnp.where(mask, epsilon_extend, epsilon).reshape((-1,1,1,1))
#
#
#     #.reshape((-1, 1, 1, 1)
#
#     # epsilon=jax.random.uniform(key4,(image.shape[0],),minval=0.95,maxval=1.05).reshape((-1, 1, 1, 1))*epsilon
#         # epsilon=epsilon*2
#     # print(label)
#
#
#
#     def adversarial_loss(perturbation):
#         logits = model(jnp.clip(image + perturbation, 0, 1))
#         # print(logits.shape,label.shape)
#         loss_value = jnp.mean(optax.softmax_cross_entropy(logits, label))
#         # loss_value = logits
#         return loss_value
#
#     # for _ in range(maxiter):
#     #     # compute gradient of the loss wrt to the image
#     #     sign_grad = jnp.sign(adversarial_loss(image_perturbation))
#
#     grad_adversarial = jax.grad(adversarial_loss)
#     metrics={}
#     prev_image_perturbation=image_perturbation
#     prev_image_perturbations=[image_perturbation]
#     for i in range(maxiter):
#
#         if dynamic:
#             pass
#             # key1, key2 = jax.random.split(key2)
#             # adv_step_size = jax.random.uniform(key2,(1,),minval=0.5,maxval=1.5).reshape((-1,1,1,1))*step_size
#             # adv_step_size = jax.random.uniform(key2, (1,), minval=0.7, maxval=1.2).reshape((-1, 1, 1, 1)) * step_size
#             # adv_step_size = jax.random.uniform(key3, (image.shape[0],), minval=0.5, maxval=1.5).reshape((-1, 1, 1, 1)) * step_size
#             # epsilon_adv = jax.random.uniform(key4, (image.shape[0],), minval=0.8, maxval=1.2).reshape(-1,1,1,1) * epsilon
#         else:
#             adv_step_size = step_size
#             # epsilon_adv=epsilon
#         # if dynamic:
#         #     key1, key2 = jax.random.split(key2)
#         #     adv_step_size = jax.random.uniform(key1, (1,), minval=0.5, maxval=1) * step_size
#
#         # compute gradient of the loss wrt to the image
#         grad=grad_adversarial(image_perturbation)
#
#         metrics[f'top_p_{i}'],factor=_nucleus_sampling(logits=einops.rearrange(grad,'b h w c -> b (h w c)'))
#
#         sign_grad = jnp.sign(grad)
#
#         sign_grad*=einops.rearrange(factor,'b (h w c)-> b h w c',h=h,w=w,c=c)
#         # sign_grad*=2/3
#         # heuristic step-size 2 eps / maxiter
#         image_perturbation += adv_step_size * sign_grad
#         # projection step onto the L-infinity ball centered at image
#         image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)
#
#
#
#         for j,prev_image_perturbation in enumerate(prev_image_perturbations):
#             delta=(prev_image_perturbation==image_perturbation).mean()
#             metrics[f'delta_{i}_{j}']=delta
#         prev_image_perturbations.append(image_perturbation)
#
#
#     # sign_grad = jnp.sign(grad_adversarial(image_perturbation))
#     # image_perturbation += step_size * sign_grad
#     # image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)
#
#     # clip the image to ensure pixels are between 0 and 1
#     image_perturbation = jnp.clip(image + image_perturbation, 0, 1)
#     return jax.lax.stop_gradient(image_perturbation),metrics


# def _nucleus_sampling(ps=[0.3,0.5,0.7], betas=[1/2,1/4,1/6], *, logits):
#
#   logits=jnp.abs(logits)
#
#   logits_sorted = jnp.sort(logits, axis=-1, descending=True)
#   sorted_cum_probs = jnp.cumsum(
#       jax.nn.softmax(logits_sorted, axis=-1), axis=-1)
#
#   # 对于每个 p，计算 cutoff logit
#   cutoff_logits = []
#   for p in ps:
#       # cutoff_index: 每个样本中，累计概率小于 p 的元素个数
#       cutoff_index = jnp.sum(sorted_cum_probs < p, axis=-1, keepdims=True)
#       # 从排序后的 logits 中取出对应位置的 cutoff logit
#       cutoff_logit = jnp.take_along_axis(logits_sorted, cutoff_index, axis=-1)
#       cutoff_logits.append(cutoff_logit)
#
#   # 初始化调整因子，默认全部为 1.0（即不下调）
#   factor = jnp.ones_like(logits)
#
#   # 为了使得“尾部”更激进，我们从最严格的 bin（ps 最后一个，对应最低 cutoff）开始赋值，
#   # 如果某个位置的 logits 小于当前 bin 的 cutoff，则赋予对应 beta，下层的覆盖前面的
#   for cutoff_logit, beta_val in zip(cutoff_logits[::-1], betas[::-1]):
#       factor = jnp.where(logits < cutoff_logit, beta_val, factor)
#
#   return factor
  # logits = jnp.where(logits < cutoff_logit,
  #                    jnp.full_like(logits, neg_inf), logits)
  # return (logits < cutoff_logit).mean(),jnp.where(logits < cutoff_logit,
  #                    jnp.full_like(logits, beta), 1.0)



# def _nucleus_sampling(ps=[0.85,0.9,0.99], betas=[3/4,1/2,1/8], *, logits):
def _nucleus_sampling(ps=[0.95,0.99], betas=[1/2,1/8], *, logits):
  logits=jnp.abs(logits)

  logits_sorted = jnp.sort(logits, axis=-1, descending=True)
  # sorted_cum_probs = jnp.cumsum(
  #     jax.nn.softmax(logits_sorted, axis=-1), axis=-1)
  sorted_cum_probs = jnp.cumsum(logits_sorted, axis=-1)
  sorted_cum_probs/=sorted_cum_probs[:,-1:]
  # print(sorted_cum_probs)


  # 对于每个 p，计算 cutoff logit
  cutoff_logits = []
  for p in ps:
      # cutoff_index: 每个样本中，累计概率小于 p 的元素个数
      cutoff_index = jnp.sum(sorted_cum_probs < p, axis=-1, keepdims=True)
      # 从排序后的 logits 中取出对应位置的 cutoff logit
      cutoff_logit = jnp.take_along_axis(logits_sorted, cutoff_index, axis=-1)
      cutoff_logits.append(cutoff_logit)

  # 初始化调整因子，默认全部为 1.0（即不下调）
  factor = jnp.ones_like(logits)

  # 为了使得“尾部”更激进，我们从最严格的 bin（ps 最后一个，对应最低 cutoff）开始赋值，
  # 如果某个位置的 logits 小于当前 bin 的 cutoff，则赋予对应 beta，下层的覆盖前面的
  for cutoff_logit, beta_val in zip(cutoff_logits, betas):
      factor = jnp.where(logits < cutoff_logit, beta_val, factor)

  return factor




def pgd_dynamic_scale_attack(image, label, model, epsilon=4 / 255, step_size=4/3 / 255, maxiter=3, key=None,dynamic=False):
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
    b,h,w,c=image.shape
    key1,key2,key3,key4,key5=jax.random.split(key,5)
    image_perturbation = jax.random.uniform(key1, image.shape, minval=-epsilon, maxval=epsilon)

    if dynamic:
        # adv_step_size=jax.random.uniform(key2,(1,),minval=0.75,maxval=1).reshape((-1,1,1,1))*step_size
        adv_step_size=step_size

    def adversarial_loss(perturbation):
        logits = model(jnp.clip(image + perturbation, 0, 1))
        # print(logits.shape,label.shape)
        loss_value = jnp.mean(optax.softmax_cross_entropy(logits, label))
        # loss_value = logits
        return loss_value

    grad_adversarial = jax.grad(adversarial_loss)
    metrics={}
    prev_image_perturbations=[image_perturbation]
    for i in range(maxiter):

        if dynamic:
            pass
        else:
            adv_step_size = step_size
        grad=grad_adversarial(image_perturbation)
        flatten_grad=einops.rearrange(grad,'b h w c -> b (h w c)')

        # metrics[f'top_p_{i}'],factor=_nucleus_sampling(logits=einops.rearrange(grad,'b h w c -> b (h w c)'))
        factor = _nucleus_sampling(logits=einops.rearrange(grad, 'b h w c -> b (h w c)'))
        # metrics[f'norm_{i}']=jnp.linalg.norm(flatten_grad,ord=1,axis=1).mean()


        sign_grad = jnp.sign(grad)

        if dynamic:
            sign_grad*=einops.rearrange(factor,'b (h w c)-> b h w c',h=h,w=w,c=c)
        # heuristic step-size 2 eps / maxiter
        image_perturbation += adv_step_size * sign_grad
        # projection step onto the L-infinity ball centered at image
        image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)



        for j,prev_image_perturbation in enumerate(prev_image_perturbations):
            delta=(prev_image_perturbation==image_perturbation).mean()
            metrics[f'delta_{i}_{j}']=delta
        prev_image_perturbations.append(image_perturbation)


    # clip the image to ensure pixels are between 0 and 1
    image_perturbation = jnp.clip(image + image_perturbation, 0, 1)
    return jax.lax.stop_gradient(image_perturbation),metrics






def pgd_attack(image, label, model, epsilon=4 / 255, step_size=4/3 / 255, maxiter=3, key=None):
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
        logits = model(jnp.clip(image + perturbation, 0, 1))
        loss_value = jnp.mean(optax.softmax_cross_entropy(logits, label))
        return loss_value

    grad_adversarial = jax.grad(adversarial_loss)
    metrics = {}
    prev_image_perturbations = [image_perturbation]
    for i in range(maxiter):

        adv_step_size = step_size
        grad = grad_adversarial(image_perturbation)
        sign_grad = jnp.sign(grad)
        # heuristic step-size 2 eps / maxiter
        image_perturbation += adv_step_size * sign_grad
        # projection step onto the L-infinity ball centered at image
        image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)

        for j, prev_image_perturbation in enumerate(prev_image_perturbations):
            delta = (prev_image_perturbation == image_perturbation).mean()
            metrics[f'delta_{i}_{j}'] = delta
        prev_image_perturbations.append(image_perturbation)

    # clip the image to ensure pixels are between 0 and 1
    image_perturbation = jnp.clip(image + image_perturbation, 0, 1)
    return jax.lax.stop_gradient(image_perturbation), metrics


def apgd_ce_attack(image, label, model, epsilon=4 / 255, step_size=None, maxiter=3, key=None):
    """PGD attack with adaptive step size (APGD-style) using CE loss.

    Args:
        image: array-like, input data for the model
        label: integer, class label corresponding to image
        model: function, the model to attack
        epsilon: float, radius of the L-infinity ball
        step_size: float, initial step size (defaults to 2*epsilon if None)
        maxiter: int, number of iterations
        key: PRNGKey for random initialization

    Returns:
        perturbed_image: Adversarial image
        metrics: dictionary of metrics about the attack
    """

    # Set initial step size if not provided
    if step_size is None:
        step_size = 2.0 * epsilon

    # Initialize metrics
    metrics = {}

    # Initialize random perturbation if key is provided
    if key is not None:
        image_perturbation = jax.random.uniform(key, image.shape, minval=-epsilon, maxval=epsilon)
    else:
        image_perturbation = jnp.zeros_like(image)

    # APGD parameters
    n_iter_check = max(int(0.1 * maxiter), 1)  # Check oscillations every 10% of iterations
    n_iter_min = max(int(0.06 * maxiter), 1)  # Minimum number of iterations between checks
    reduced_last_check = jnp.ones(image.shape[0])  # Track if step size was reduced in last check
    thr_decr = 0.75  # Threshold for oscillation detection

    # To track best results
    best_perturbation = image_perturbation
    best_loss = jnp.ones(image.shape[0]) * (-float('inf'))
    loss_steps = jnp.zeros((maxiter, image.shape[0]))

    def cross_entropy_loss(perturbation):
        """CE loss function for generating adversarial examples."""
        logits = model(jnp.clip(image + perturbation, 0, 1))
        loss_value = -optax.softmax_cross_entropy(logits, label)  # Negative CE to maximize
        return loss_value

    # Get gradient function
    grad_fn = jax.grad(cross_entropy_loss, has_aux=False)

    # Initialize step size for each example
    step_sizes = jnp.ones((image.shape[0], 1, 1, 1)) * step_size

    # Previous perturbation for momentum
    prev_perturbation = image_perturbation

    # Main attack loop
    for i in range(maxiter):
        # Calculate loss and gradient
        loss = cross_entropy_loss(image_perturbation)
        grad = grad_fn(image_perturbation)

        # Update loss tracking
        loss_steps = loss_steps.at[i].set(loss)

        # Check if current loss is better than best loss
        is_better = loss > best_loss
        best_loss = jnp.where(is_better, loss, best_loss)
        best_perturbation = jnp.where(is_better.reshape(-1, 1, 1, 1),
                                      image_perturbation,
                                      best_perturbation)

        # Gradient momentum (as in APGD)
        if i > 0:
            # Calculate grad2 (like in APGD)
            grad2 = image_perturbation - prev_perturbation

            # APGD uses momentum parameter a = 0.75 for iteration > 0
            a = 0.75

            # Update with sign of gradient and momentum
            new_perturbation = image_perturbation + step_sizes * jnp.sign(grad)
            # Project to epsilon ball
            new_perturbation = jnp.clip(new_perturbation, -epsilon, epsilon)
            # Apply momentum
            image_perturbation = image_perturbation + (new_perturbation - image_perturbation) * a + grad2 * (1 - a)
            image_perturbation = jnp.clip(image_perturbation, -epsilon, epsilon)
        else:
            # For first iteration, simple gradient step
            image_perturbation = image_perturbation + step_sizes * jnp.sign(grad)
            image_perturbation = jnp.clip(image_perturbation, -epsilon, epsilon)

        # Save current perturbation for next iteration's momentum
        prev_perturbation = image_perturbation

        # Check for oscillations and adjust step size (APGD adaptive mechanism)
        if (i + 1) % n_iter_check == 0 and i > 0:
            # Check for oscillations in last n_iter_check iterations
            losses_to_check = loss_steps[i - n_iter_check + 1:i + 1]

            # Implementation of check_oscillation function
            def check_oscillation(losses):
                # Count how many times the loss increased
                t = jnp.sum(losses[1:] > losses[:-1], axis=0)
                # If less than 75% of iterations show increase, we have oscillation
                return (t <= n_iter_check * thr_decr).astype(jnp.float32)

            fl_oscillation = check_oscillation(losses_to_check)

            # Also check if no improvement since last check
            fl_reduce_no_impr = (1. - reduced_last_check) * (best_loss <= loss)
            fl_oscillation = jnp.maximum(fl_oscillation, fl_reduce_no_impr)
            reduced_last_check = fl_oscillation

            # Reduce step size where oscillation detected
            step_sizes = jnp.where(fl_oscillation.reshape(-1, 1, 1, 1),
                                   step_sizes * 0.5,
                                   step_sizes)

            # Reset to best perturbation where oscillation detected
            image_perturbation = jnp.where(fl_oscillation.reshape(-1, 1, 1, 1),
                                           best_perturbation,
                                           image_perturbation)

            # Update n_iter_check to check less frequently
            n_iter_check = max(n_iter_check - 1, n_iter_min)

            # Record step size changes
            metrics[f'step_size_i{i}'] = step_sizes.mean().item()

    # Return final perturbed image
    perturbed_image = jnp.clip(image + best_perturbation, 0, 1)
    return jax.lax.stop_gradient(perturbed_image), metrics




# def pgd_attack(image, label, model, epsilon=4 / 255, step_size=4/3 / 255, maxiter=3, key=None):
#     """PGD attack on the L-infinity ball with radius epsilon.
#
#   Args:
#     image: array-like, input data for the CNN
#     label: integer, class label corresponding to image
#     params: tree, parameters of the model to attack
#     epsilon: float, radius of the L-infinity ball.
#     maxiter: int, number of iterations of this algorithm.
#
#   Returns:
#     perturbed_image: Adversarial image on the boundary of the L-infinity ball
#       of radius epsilon and centered at image.
#
#   Notes:
#     PGD attack is described in (Madry et al. 2017),
#     https://arxiv.org/pdf/1706.06083.pdf
#     :param state:
#     :param image:
#     :param label:
#     :param params:
#     :param maxiter:
#     :param epsilon:
#     :param step_size:
#   """
#
#     # image = einops.rearrange(image, 'b c h w->b h w c')
#     # image = image.astype(jnp.float32)
#     # label = label.astype(jnp.int32)
#
#     # image_perturbation = jnp.zeros_like(image)
#     image_perturbation = jax.random.uniform(key, image.shape, minval=-epsilon, maxval=epsilon)
#
#     # print(label)
#
#     def adversarial_loss(perturbation):
#         logits = model(jnp.clip(image + perturbation, 0, 1))
#         # print(logits.shape,label.shape)
#         loss_value = jnp.mean(optax.softmax_cross_entropy(logits, label))
#         # loss_value = logits
#         return loss_value
#
#     # for _ in range(maxiter):
#     #     # compute gradient of the loss wrt to the image
#     #     sign_grad = jnp.sign(adversarial_loss(image_perturbation))
#
#     grad_adversarial = jax.grad(adversarial_loss)
#     # for _ in range(maxiter):
#     #     # compute gradient of the loss wrt to the image
#     #     sign_grad = jnp.sign(grad_adversarial(image_perturbation))
#     #
#     #     # heuristic step-size 2 eps / maxiter
#     #     image_perturbation += step_size * sign_grad
#     #     # projection step onto the L-infinity ball centered at image
#     #     image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)
#
#
#     def loop_body(i,image_perturbation):
#         sign_grad = jnp.sign(grad_adversarial(image_perturbation))
#
#         # heuristic step-size 2 eps / maxiter
#         image_perturbation += step_size * sign_grad
#         # projection step onto the L-infinity ball centered at image
#         image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)
#         return image_perturbation
#
#
#     image_perturbation=jax.lax.fori_loop(0,maxiter,loop_body,init_val=image_perturbation)
#
#     # clip the image to ensure pixels are between 0 and 1
#     image_perturbation = jnp.clip(image + image_perturbation, 0, 1)
#     return jax.lax.stop_gradient(image_perturbation)




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



def rgd_dynamic_scale_attack(image, label, model, epsilon=4 / 255, step_size=4/3 / 255, maxiter=3, key=None,dynamic=False):
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
    b,h,w,c=image.shape
    key1,key2,key3,key4,key5=jax.random.split(key,5)
    image_perturbation = jax.random.uniform(key1, image.shape, minval=-epsilon, maxval=epsilon)

    if dynamic:
        # adv_step_size=jax.random.uniform(key2,(1,),minval=0.75,maxval=1).reshape((-1,1,1,1))*step_size
        adv_step_size=step_size

    def adversarial_loss(perturbation):
        logits = model(jnp.clip(image + perturbation, 0, 1))
        # print(logits.shape,label.shape)
        loss_value = jnp.mean(optax.softmax_cross_entropy(logits, label))
        # loss_value = logits
        return loss_value

    grad_adversarial = jax.grad(adversarial_loss)
    metrics={}
    prev_image_perturbations=[image_perturbation]
    for i in range(maxiter):


        grad=grad_adversarial(image_perturbation)

        if dynamic:
            sign_grad = grad
            adv_step_size = 1e4
        else:
            adv_step_size = step_size
            sign_grad = jnp.sign(grad)


        # heuristic step-size 2 eps / maxiter
        image_perturbation += adv_step_size * sign_grad
        # projection step onto the L-infinity ball centered at image



        for j,prev_image_perturbation in enumerate(prev_image_perturbations):
            delta=(prev_image_perturbation==image_perturbation).mean()
            metrics[f'delta_{i}_{j}']=delta
        prev_image_perturbations.append(image_perturbation)

        if dynamic:
            pass
        else:
            image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)

    image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)


    # clip the image to ensure pixels are between 0 and 1
    image_perturbation = jnp.clip(image + image_perturbation, 0, 1)
    return jax.lax.stop_gradient(image_perturbation),metrics

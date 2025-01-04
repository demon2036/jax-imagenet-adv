"""Muon.
Implementation of the
[Muon optimizer](https://github.com/KellerJordan/modded-nanogpt)
by Keller Jordan
"""
import collections
import re
from typing import Any, List, NamedTuple, Optional, Tuple, Union, Callable, Sequence, Iterator

import chex
import flax
import jax
import jax.numpy as jnp
import optax
import regex
from jax._src.util import safe_map

from optax import tree_utils as otu
from optax._src import base, wrappers
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src.transform import ScaleByAdamState
import  optax._src.utils as utils

# from optax._src import utils




def scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:
  r"""Rescale updates according to the Adam algorithm.

  See :func:`optax.adam` for more details.

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    nesterov: Whether to use Nesterov momentum. The variant of Adam with
      Nesterov momentum is described in [Dozat 2016]

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = otu.tree_zeros_like(params, dtype=mu_dtype)  # First moment
    nu = otu.tree_zeros_like(params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = otu.tree_update_moment(updates, state.mu, b1, 1)
    nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_increment(state.count)
    if nesterov:
      mu_hat = jax.tree.map(
          lambda m, g: b1 * m + (1 - b1) * g,
          otu.tree_bias_correction(mu, b1, numerics.safe_increment(count_inc)),
          otu.tree_bias_correction(updates, b1, count_inc),
      )
    else:
      mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
    # Dozat 2016 https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
    # Algorithm 2 further multiplies Adam's standard nu_hat by b2. It is
    # unclear why. Other Nadam implementations also omit the extra b2 factor.
    nu_hat = otu.tree_bias_correction(nu, b2, count_inc)


    # def adam_fn(v,u,g):
    #     if v is None:
    #         return None
    #
    #
    #     rms = jnp.sqrt(jnp.mean(jnp.square(g**2/ (u+eps )  ), ) +eps )
    #     scale=1/jnp.maximum(1,rms)
    #     return scale*v/(jnp.sqrt(u + eps_root) + eps)
    #
    #
    # updates = jax.tree.map(
    #     adam_fn,
    #     mu_hat,
    #     nu_hat,updates,
    #     is_leaf=lambda x: x is None,
    # )

    def get_scale(x,v):
        rms=jnp.sqrt(jnp.mean(jnp.square(x**2/ (v+eps )  ), ) +eps )
        return 1/rms

    scale=jax.tree_util.tree_map(get_scale,updates,nu_hat)


    updates = jax.tree.map(
        lambda m, v: None if m is None else m / (jnp.sqrt(v + eps_root) + eps),
        mu_hat,
        nu_hat,
        is_leaf=lambda x: x is None,
    )



    mu = otu.tree_cast(mu, mu_dtype)
    return (updates,scale), ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)





def add_decayed_weights(
    weight_decay: Union[float, jax.Array] = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  """Add parameter scaled by `weight_decay`.

  Args:
    weight_decay: A scalar weight decay rate.
    mask: A tree with same structure as (or a prefix of) the params PyTree, or a
      Callable that returns such a pytree given the params/updates. The leaves
      should be booleans, `True` for leaves/subtrees you want to apply the
      transformation to, and `False` for those you want to skip.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def update_fn(carry, state, params):

    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)

    updates, scale=carry

    updates = jax.tree.map(
        lambda g, p,s: None if g is None else s*(g + weight_decay * p),
        updates,
        params,scale,
        is_leaf=lambda x: x is None,
    )
    return updates, state

  # If mask is not `None`, apply mask to the gradient transformation.
  # E.g. it is common to skip weight decay on bias units and batch stats.
  if mask is not None:
    return wrappers.masked(
        base.GradientTransformation(base.init_empty_state, update_fn), mask
    )
  return base.GradientTransformation(base.init_empty_state, update_fn)


def stable_adamw(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:

# transform.scale_by_adam()
  return combine.chain(
      scale_by_adam(
          b1=b1,
          b2=b2,
          eps=eps,
          eps_root=eps_root,
          mu_dtype=mu_dtype,
          nesterov=nesterov,
      ),
      add_decayed_weights(weight_decay, mask),
      transform.scale_by_learning_rate(learning_rate),
  )






def modified_lamb2(
        learning_rate: optax.ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-7,
        eps_root: float = 0.0,
        weight_decay: float = 0.0,
        mask: optax.MaskOrFn = None,
) -> optax.GradientTransformation:
    return optax.chain(
        scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
        add_decayed_weights(weight_decay=weight_decay, mask=mask),
        # Change to use trust ratio on weight decay parameters only.
        optax.masked(optax.scale_by_trust_ratio(), mask=mask),
        optax.scale_by_learning_rate(learning_rate),
    )
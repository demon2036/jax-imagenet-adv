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
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
# from optax._src import utils


class SafeZipIteratorError(RuntimeError):
  pass

class SafeZipIterator:
  """Lazy zip over multiple iterators, ensuring that all have the same length."""

  def __init__(self, *iterators):
    self.iterators = tuple(
        i if isinstance(i, collections.abc.Iterator) else iter(i)
        for i in iterators)

  def __iter__(self):
    return self

  def __next__(self) -> Tuple[Any, ...]:
    stop = None
    elements = []
    for i, iterator in enumerate(self.iterators):
      try:
        elements.append(next(iterator))
        if stop is not None:
          break
      except StopIteration:
        stop = i
    if stop is not None and elements:
      raise SafeZipIteratorError(
          f'The {stop}-th iterator raised StopIteration before the rest')
    if not elements:
      raise StopIteration
    return tuple(elements)

def safe_map(f: Callable[..., Any], *iterables) -> Iterator[Any]:
  for args in SafeZipIterator(*iterables):
    yield f(*args)


WeightDecay = Union[float, Sequence[Tuple[str, float]]]

def add_decayed_weights(
    weight_decay: Optional[WeightDecay],mask=None) -> optax.GradientTransformation:
  """Optionally adds parameters scaled by weight_decay.

  This is similar to `optax.add_decayed_weights`, but supports passing different
  weight decay factors for each parameter, by matching the parameter names with
  a regex. The factor of the first regex matched will be used.

  Examples:
    add_decayed_weights(1e-3)   # All parameters are decayed with factor = 1e-3.

    add_decayed_weights([
      ('Head/kernel', 0.1),         # Head kernel uses factor = 0.1.
      ('Encoder/*/kernel', 0.001),  # Kernels in the encoder factor = 0.001.
      # Parameters that do not match the above regexes are not decayed.
    ])

  Args:
    weight_decay: A float or a mapping from regexes to floats.

  Returns:
    An optax.GradientTransformation object.
  """
  print(weight_decay)
  if not weight_decay:
    return optax.identity()
  elif isinstance(weight_decay, (list, tuple)):
    weight_decay = [(re.compile(k), v) for k, v in weight_decay]
    def weight_decay_fn(key):
      for regex, value in weight_decay:
        if regex.search(key):
            return value
      return 0.
  else:
    def weight_decay_fn(unused_key):
      return weight_decay

  def init_fn(_):
    return optax.AddDecayedWeightsState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError('Not passing `params` when calling `update`.')
    flatupdates = flax.traverse_util.flatten_dict(
        flax.serialization.to_state_dict(updates), sep='/')
    flatparams = flax.traverse_util.flatten_dict(
        flax.serialization.to_state_dict(params), sep='/')
    flatupdates = dict(safe_map(
        lambda k, g, p: (k, g + weight_decay_fn(k) * p),
        flatupdates.keys(),
        flatupdates.values(),
        flatparams.values()))
    updates = flax.serialization.from_state_dict(
        updates, flax.traverse_util.unflatten_dict(flatupdates, sep='/'))
    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)





"""
def add_decayed_weights(
    weight_decay: Union[float, jax.Array] = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:


  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    updates = jax.tree.map(
        lambda g, p: None if g is None else g + weight_decay * p,
        updates,
        params,
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

"""




def modified_lion(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.99,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-3,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:

  return combine.chain(
      transform.scale_by_lion(b1=b1, b2=b2, mu_dtype=mu_dtype),
      add_decayed_weights(weight_decay, mask),
      transform.scale_by_learning_rate(learning_rate),
  )

if __name__=="__main__":
    v='model/MetaFormerStage_1/MetaFormerBlock_7/mlp/fc1/kernel'

    p=[['.*/fc/.*/kernel', 1.0], ['.*/kernel', 0.01]]

    print(regex.search(regex.compile(p[1][0]),v))




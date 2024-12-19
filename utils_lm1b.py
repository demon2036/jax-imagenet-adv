import functools
from typing import Callable

import flax.linen as nn
import flax.traverse_util
import jax
import jax.numpy as jnp
import os

import optax
from flax.training import train_state
from jax.lax import with_sharding_constraint
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
jax.config.update('jax_platform_name', 'cpu')

device_mesh = mesh_utils.create_device_mesh((2, 4))
print(device_mesh)

mesh = Mesh(devices=device_mesh, axis_names=('data', 'model'))
print(mesh)

def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
  return NamedSharding(mesh, pspec)




class LogicalDotReluDot(nn.Module):
  depth: int
  dense_init: Callable = nn.initializers.xavier_normal()
  @nn.compact
  def __call__(self, x):
    y = nn.Dense(self.depth,
                 kernel_init=nn.with_logical_partitioning(self.dense_init, ('embed', 'hidden')),
                 use_bias=False,  # or overwrite with `bias_init`
                 )(x)

    y = jax.nn.relu(y)
    # Force a local sharding annotation.
    y = with_sharding_constraint(y, mesh_sharding(PartitionSpec('data', 'model')))

    W2 = self.param(
        'W2',
        nn.with_logical_partitioning(self.dense_init, ('hidden', 'embed')),
        (self.depth, x.shape[-1]))

    z = jnp.dot(y, W2)
    # Force a local sharding annotation.
    z = nn.with_logical_constraint(z, ('batch', 'embed'))
    return z, None

class LogicalMLP(nn.Module):
  num_layers: int
  depth: int
  use_scan: bool
  @nn.compact
  def __call__(self, x):
    if self.use_scan:
      x, _ = nn.scan(LogicalDotReluDot, length=self.num_layers,
                    variable_axes={"params": 0},
                    split_rngs={"params": True},
                    metadata_params={nn.PARTITION_NAME: 'layer'}
                    )(self.depth)(x)
    else:
      for i in range(self.num_layers):
        x, _ = LogicalDotReluDot(self.depth)(x)
    return x

# MLP hyperparameters.
BATCH, LAYERS, DEPTH, USE_SCAN = 8, 4, 1024, False
# Create fake inputs.
x = jnp.ones((BATCH, DEPTH))
# Initialize a PRNG key.
k = jax.random.key(0)

# Create an Optax optimizer.
optimizer = optax.adam(learning_rate=0.001)
# Instantiate the model.
logical_model = LogicalMLP(LAYERS, DEPTH, USE_SCAN)


rules = (('batch', 'data'),
         ('hidden', 'model'))


def init_fn(k, x, model, optimizer):
  variables = model.init(k, x) # Initialize the model.
  state = train_state.TrainState.create( # Create a `TrainState`.
    apply_fn=model.apply,
    params=variables['params'],
    tx=optimizer)
  return state

logical_abstract_variables = jax.eval_shape(
    functools.partial(init_fn, model=logical_model, optimizer=optimizer), k, x)

logical_state_spec = nn.get_partition_spec(logical_abstract_variables)
def pp(p,x):
    print(p,x.sharding,)

# print(logical_abstract_variables.params)
print(logical_state_spec)
jax.tree_util.tree_map_with_path(pp,flax.traverse_util.flatten_dict(logical_abstract_variables.params,sep='/'))

logical_state_sharding = nn.logical_to_mesh_sharding(logical_state_spec, mesh, rules)
print(logical_state_sharding)
x_sharding = mesh_sharding(PartitionSpec('data', None)) # dimensions: (batch, length)
x = jax.device_put(x, x_sharding)

logical_jit_init_fn = jax.jit(init_fn, static_argnums=(2, 3),
                      in_shardings=(mesh_sharding(()), x_sharding),  # PRNG key and x
                      out_shardings=logical_state_sharding)

logical_initialized_state = logical_jit_init_fn(k, x, logical_model, optimizer)
jax.tree_util.tree_map_with_path(pp,logical_initialized_state)


import jax.numpy as jnp
import jax.random




a=jnp.ones((2,2))
b=jnp.ones((2,2))+3
arr1_expanded = a[None, :]  # Shape becomes (1, 3)
arr2_expanded = b[None, :]  # Shape becomes (1, 3)

# Concatenate along the new axis (axis 0)
result = jnp.concatenate([arr1_expanded, arr2_expanded], axis=0)


print(jax.random.choice(jax.random.PRNGKey(1),result,p=jnp.array([0.9,0.5])))
import jax

jax.distributed.shutdown()
jax.distributed.initialize()
print(1)
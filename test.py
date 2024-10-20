import jax
print(jax.process_index())
jax.distributed.shutdown()
jax.distributed.initialize()
print(1)
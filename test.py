import jax

jax.print_environment_info()

jax.distributed.initialize(initialization_timeout=10000)
# print(jax.devices())
print(1)
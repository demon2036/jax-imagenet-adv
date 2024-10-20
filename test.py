import jax
print(jax.process_index())

with open('test.txt','w') as f:
    f.write(str(jax.process_index()))


jax.distributed.shutdown()
jax.distributed.initialize()

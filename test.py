import time
from os import times

import jax



import timm

jax.distributed.initialize()
for i in range(10):
    time.sleep(1)
    print(i)

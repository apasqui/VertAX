import jax.numpy as jnp

a = jnp.arange(20).reshape((4, 5))
b = jnp.arange(5)
c = jnp.arange(4)

from functools import partial
from jax import vmap

def f(a, b):
  return a + b, 2 * b

d, e = vmap(f, in_axes=(0, None))(a, b)
print(e)
print(d)
d = d.ravel()

print(d)
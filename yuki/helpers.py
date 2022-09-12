import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from functools import partial

default = lambda x, backup : x if x is not None else backup

def RNG(old):
    while True:
        old, new = jr.split(old)
        yield new
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import equinox as nox
from functools import partial

def RNG(old):
    while True:
        old, new = jr.split(old)
        yield new


def batch(f):
    batch = partial(nox.filter_vmap, kwargs={"self":None})

    def inner(self, x, *args, **kwargs):
        b, *_ = x.shape
        if (key := kwargs.get("key")) is not None:
            kwargs["key"] = jr.split(key,b)
        out = batch(f)(self, x, *args, **kwargs)

        return out

    return inner

def replicate(model):
    dynamic, static = nox.partition(model, nox.is_array)
    pdynamic = jax.device_put_replicated(dynamic, jax.local_devices())
    pmodel = nox.combine(pdynamic, static)

    return pmodel

def unreplicate(pmodel):
    pdynamic, static = nox.partition(pmodel, nox.is_array)
    dynamic = jtu.tree_map(lambda x : x[0], pdynamic)
    model = nox.combine(dynamic, static)

    return model
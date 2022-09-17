import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

import numpy as onp
import equinox as eqx
from functools import partial

def RNG(old):
    while True:
        old, new = jr.split(old)
        yield new

def batch(f):
    batch = partial(eqx.filter_vmap, kwargs={"self":None})

    def inner(self, x, *args, **kwargs):
        b, *_ = x.shape
        if (key := kwargs.get("key")) is not None:
            kwargs["key"] = jr.split(key,b)
        out = batch(f)(self, x, *args, **kwargs)

        return out

    return inner

def replicate(model):
    dynamic, static = eqx.partition(model, eqx.is_array)
    pdynamic = jax.device_put_replicated(dynamic, jax.local_devices())
    pmodel = eqx.combine(pdynamic, static)

    return pmodel

def unreplicate(pmodel, axis = 0):
    pdynamic, static = eqx.partition(pmodel, eqx.is_array)
    dynamic = jtu.tree_map(lambda x : x[axis], pdynamic)
    model = eqx.combine(dynamic, static)

    return model

def tensor_to_image(x) -> onp.ndarray:
    x = x * 0.5 + 0.5 # [-1 ~ 1] -> [0 ~ 1]
    x = onp.uint8(x.clip(0,1) * 255)
    return x

def L2(x, axis=None):
    return jnp.sqrt(jnp.sum(x ** 2, axis=axis))

def normalise(x, axis=-1, eps=1e-8):
    return x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=axis, keepdims=True) + eps)

def convolve(x, w, stride=(1,1), padding=((0,0),(0,0)), groups=1, flip=True, transpose=False):
    kwargs = { "feature_group_count":groups, "dimension_numbers":("NCHW", "OIHW", "NCHW") }
    if not flip: w = jnp.flip(w, axis=(2,3))

    if transpose:
        _, _, kh, kw = w.shape
        w = jnp.flip(w, axis=(2,3))
        w = jnp.swapaxes(w, 0, 1)

        (pxmin, pxmax), (pymin, pymax) = padding
        pxmin, pxmax = (kh - 1) - pxmin, (kh - 1) - pxmax
        pymin, pymax = (kw - 1) - pymin, (kw - 1) - pymax

        return jax.lax.conv_general_dilated(x, w, (1,1), ((pxmin, pxmax), (pymin, pymax)), stride, **kwargs)
    else:
        return jax.lax.conv_general_dilated(x, w, stride, padding, **kwargs)
    
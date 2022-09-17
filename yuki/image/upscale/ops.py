import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn.initializers as jinit
from equinox import nn, static_field, Module
from einops import rearrange, reduce
from yuki.common.ops import convolve

def RNG(old):
    while True:
        new, old = jr.split(old)
        yield new

class LReLU(Module):
    negative_slope:float = static_field()

    def __init__(self, negative_slope:float = 0.2):
        self.negative_slope = negative_slope

    def __call__(self, x, key=None):
        return jax.nn.leaky_relu(x, negative_slope=self.negative_slope)

class Upscale(Module):
    scale:int = static_field()

    def __init__(self, scale:int):
        self.scale = scale

    def __call__(self, x, key = None):
        c, h, w = x.shape
        shape = (c, h*self.scale, w * self.scale)
        return jax.image.resize(x, shape, method="nearest")

class Deconvolution(Module):
    weight:jnp.ndarray
    bias:jnp.ndarray
    stride:tuple = static_field()
    padding:tuple = static_field()
    dilation:tuple = static_field()
    use_bias:bool = static_field()

    def __init__(
        self, in_features, out_features, kernel, 
        stride = (1,1), padding = (0,0), dilation=(1,1), 
        use_bias = False, dtype=jnp.float32, key = None
    ):
        wkey, bkey = jr.split(key)
        uniform, normal = jinit.uniform(), jinit.he_normal()
        weight = normal(wkey, (out_features, in_features, *kernel), dtype=dtype)
        bias = uniform(bkey, [out_features], dtype=dtype)
        self.weight = rearrange(weight, 'o i kh kw -> i o kh kw')
        self.bias = rearrange(bias, 'o -> o 1 1')

        self.stride = stride
        self.padding = [padding, padding]
        self.dilation = dilation
        self.use_bias = use_bias

    def __call__(self, x, key = None):
        x = rearrange(x, 'c h w -> 1 c h w')

        x = convolve(x, self.weight, self.stride, self.padding, self.dilation, transpose = True)
        x = rearrange(x, 'n c h w -> (n c) h w')

        if self.use_bias:
            x += self.bias

        return x
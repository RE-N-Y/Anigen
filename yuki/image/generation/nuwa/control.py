import jax
import jax.numpy as jnp
import jax.random as jr

import numpy as onp
import equinox
from equinox import nn, static_field, Module
from einops import rearrange

# ADC.split

class ADC(Module):
    height:int = static_field()
    width:int = static_field()

    def __init__(self, height:int, width:int):
        self.height, self.width = height, width

    def split(self, image, order:str = "zeta", reverse = False):
        if order == "zeta" : return self.zeta(image, reverse = reverse)
        if order == "omega": return self.omega(image, reverse = reverse)
        raise ValueError(f"{order} should be one of zeta / omega")

    def compose(self, patches, order:str = "zeta", reverse = False):
        if order == "zeta" : return self.rzeta(patches, reverse = reverse)
        if order == "omega": return self.romega(patches, reverse = reverse)
        raise ValueError(f"{order} should be one of zeta / omega")

    def zeta(self, x, reverse = False):
        if reverse: x = jnp.flipud(x)
        return rearrange(x, '... h w -> ... (h w)')

    def rzeta(self, x, reverse = False):
        x = rearrange(x, '... (h w) -> h w', h = self.height, w = self.width)
        return jnp.flipud(x) if reverse else x

    def omega(self, x, reverse = False):
        if reverse: x = jnp.fliplr(x)
        return rearrange(x, '... h w -> ... (w h)')

    def romega(self, x, reverse = False):
        x = rearrange(x, '... (w h) -> ... h w', h = self.height, w = self.width)
        return jnp.fliplr(x) if reverse else x

# TODO: Hilbert curves

class ADCEmbedding(Module):
    embedding:nn.Embedding

    def __init__(self, features:int, key = None):
        self.embedding = nn.Embedding(18, features, key = key)

    def __call__(self, patch, context):
        return self.embedding(patch), self.embedding(context)


class NCP:
    def __init__(self, extent:tuple[int], height:int, width:int, order="zeta"):
        self.extent = extent
        self.height = height
        self.width = width
        self.adc = ADC(height, width)

        xv, yv = onp.meshgrid(onp.arange(width), onp.arange(height))
        coordinates = onp.stack((xv,yv), axis = 0)

        self.itoc = self.adc.split(coordinates)
        self.ctoi = self.adc.compose(onp.arange(height * width))


    def select(self, idx:int, pool:set):
        x,y = self.itoc[idx]
        w,h = self.extent
        
        idxes = self.ctoi[max(0, x - w) : min(self.width, x + w), max(0, y - h) : min(self.height, y + h)]
        idxes = filter(lambda i : i in pool, idxes)

        return set(idxes)

    def should_remove(self, idx:int):
        next_tokens = set()
        next_tokens = next_tokens.union(self.selections[idx:])
        
        return idx in next_tokens

    # TODO: improve runtime complexity of remove operation
    def add(self, idx:int, pool:set):
        pool = [ *pool, idx ]
        pool = filter(lambda i : self.should_remove(i), pool) # remove tokens

        return set(pool)
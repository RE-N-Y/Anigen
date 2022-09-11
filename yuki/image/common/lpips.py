import jax
import jax.numpy as jnp
import jax.random as jr

import flax
import flax.linen as nn
import flaxmodels as fm
import functools as ft
from einops import rearrange, reduce, repeat

class VGGFeatures(nn.Module):

    def setup(self):
        pooling = ft.partial(nn.max_pool, strides=(2,2), window_shape = (2,2))
        self.features = [
            nn.Conv(64,  (3,3), padding=1), nn.relu,
            nn.Conv(64,  (3,3), padding=1), nn.relu, pooling,
            nn.Conv(128, (3,3), padding=1), nn.relu,
            nn.Conv(128, (3,3), padding=1), nn.relu, pooling,
            nn.Conv(256, (3,3), padding=1), nn.relu,
            nn.Conv(256, (3,3), padding=1), nn.relu,
            nn.Conv(256, (3,3), padding=1), nn.relu, pooling,
            nn.Conv(512, (3,3), padding=1), nn.relu,
            nn.Conv(512, (3,3), padding=1), nn.relu,
            nn.Conv(512, (3,3), padding=1), nn.relu, pooling,
            nn.Conv(512, (3,3), padding=1), nn.relu,
            nn.Conv(512, (3,3), padding=1), nn.relu,
            nn.Conv(512, (3,3), padding=1), nn.relu
        ]

    def __call__(self, x, slices):
        features = []
        for layer in self.features:
            x = layer(x)
            features.append(x)

        return x, [features[idx] for idx in slices]

def normalise(x, axis=-1, eps=1e-10):
    r = jnp.sqrt(jnp.sum(x ** 2, axis=axis, keepdims=True)) + eps
    return x / r

class LPIPS(nn.Module):

    @nn.compact
    def __call__(self, x, y, slices = [3, 8, 15, 22, 29]):
        features = {}
        difference = 0.

        # scale x, y
        mean, std = jnp.array([-.030,-.088,-.188]), jnp.array([.458,.448,.450])
        x, y = x - mean, y - mean
        x, y = x / std, y / std

        backbone = VGGFeatures()
        _, features['x'] = backbone(x, slices)
        _, features['y'] = backbone(y, slices)
        
        for x, y in zip(features['x'], features['y']):
            d = normalise(x) - normalise(y)
            d = nn.Conv(1, (1,1), use_bias = False)(d ** 2)
            d = reduce(d, 'b h w c -> b 1 1 c', 'mean')
            difference += d

        return difference
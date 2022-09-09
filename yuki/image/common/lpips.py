import jax
import jax.numpy as jnp
import equinox
from equinox import nn, static_field, Module
from einops import rearrange, reduce
from yuki.helpers import RNG, batch

ReLU = lambda : nn.Lambda(jax.nn.relu)
LReLU = lambda : nn.Lambda(jax.nn.leaky_relu)

class VGGFeatures(Module):
    features:list

    def __init__(self, *, key):
        key = RNG(key)
        self.features = [
            nn.Conv2d(3, 64, 3, padding=1, key=next(key)), ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, key=next(key)), ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1, key=next(key)), ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, key=next(key)), ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, padding=1, key=next(key)), ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, key=next(key)), ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, key=next(key)), ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(256, 512, 3, padding=1, key=next(key)), ReLU(),
            nn.Conv2d(512, 512, 3, padding=1, key=next(key)), ReLU(),
            nn.Conv2d(512, 512, 3, padding=1, key=next(key)), ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(512, 512, 3, padding=1, key=next(key)), ReLU(),
            nn.Conv2d(512, 512, 3, padding=1, key=next(key)), ReLU(),
            nn.Conv2d(512, 512, 3, padding=1, key=next(key)), ReLU()
        ]

    def __call__(self, x, slices = [-1], key=None):
        features = []
        for block in self.features:
            x = block(x)
            features.append(x)

        return x, [features[idx] for idx in slices]

def normalise(x, axis=0, eps=1e-10):
    r = jnp.sqrt(jnp.sum(x ** 2, axis=axis, keepdims=True)) + eps
    return x / r

class LPIPS(Module):
    backbone:Module
    linears:list
    slices:list = static_field()
    mean:jnp.ndarray = static_field()
    std:jnp.ndarray = static_field()

    def __init__(self,
        slices = [3, 8, 15, 22, 29],
        intermediate_features = [64, 128, 256, 512, 512],
        key=None
    ):
        key = RNG(key)
        assert len(slices) == len(intermediate_features)

        self.slices = slices
        self.backbone = VGGFeatures(key=next(key))

        mean, std = jnp.array([-.030,-.088,-.188]), jnp.array([.458,.448,.450])
        self.mean, self.std = map(lambda t : t[:,None,None], (mean, std))

        self.linears = [ nn.Conv2d(features, 1, 1, use_bias=False, key=next(key)) for features in intermediate_features ]

    @batch
    def __call__(self, x, y):
        features = {}
        difference = 0.

        # scale x, y
        x, y = x - self.mean, y - self.mean
        x, y = x / self.std, y / self.std

        _, features['x'] = self.backbone(x, self.slices)
        _, features['y'] = self.backbone(y, self.slices)

        for idx, (x, y) in enumerate(zip(features['x'], features['y'])):
            d = normalise(x, axis = 0) - normalise(y, axis = 0)
            d = reduce(self.linears[idx](d ** 2), 'c h w -> c 1 1', 'mean')
            difference += d

        return difference
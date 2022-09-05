import jax
import jax.numpy as jnp
from equinox import nn, Module
from ops import RNG, LReLU, Deconvolution

class Waifu2x(Module):
    net: Module

    def __init__(self, in_channels:int = 3, features:int = 16, key = None):
        key = RNG(key)
        self.net = nn.Sequential([
            nn.Conv2d(in_channels, features, (3,3), key=next(key)), LReLU(0.1), 
            nn.Conv2d(features, features * 2, (3,3), key=next(key)), LReLU(0.1),
            nn.Conv2d(features * 2, features * 4, (3, 3), key=next(key)), LReLU(0.1),
            nn.Conv2d(features * 4, features * 8, (3, 3), key=next(key)), LReLU(0.1),
            nn.Conv2d(features * 8, features * 8, (3, 3), key=next(key)), LReLU(0.1),
            nn.Conv2d(features * 8, features * 16, (3, 3), key=next(key)), LReLU(0.1),  
            Deconvolution(features * 16, 3, (4,4), stride=(2,2), padding=(3,3), key=next(key))
        ])

    def __call__(self, x, pad:int = 7):
        c, h, w = x.shape
        out = jnp.zeros((c, pad + h + pad, pad + w + pad))
        out = out.at[:, pad : h + pad, pad : w + pad].set(x)
        out = self.net(out)
        
        return out
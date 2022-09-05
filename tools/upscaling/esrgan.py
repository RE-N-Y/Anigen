import jax
import jax.numpy as jnp
from equinox import nn, Module
from ops import RNG, LReLU, Upscale

class ResidualDenseBlock(Module):
    layers:list
    activations:list

    def __init__(self, features:int, grow_features:int, key = None):
        key = RNG(key)
        self.activations = [LReLU(0.2), LReLU(0.2), LReLU(0.2), LReLU(0.2), nn.Identity()]
        self.layers = [
            nn.Conv2d(features, grow_features, (3, 3), padding=(1,1), key=next(key)),
            nn.Conv2d(features + grow_features, grow_features, (3, 3), padding=(1,1), key=next(key)),
            nn.Conv2d(features + 2 * grow_features, grow_features, (3,3), padding=(1,1), key=next(key)),
            nn.Conv2d(features + 3 * grow_features, grow_features, (3,3), padding=(1,1), key=next(key)),
            nn.Conv2d(features + 4 * grow_features, features, (3,3), padding=(1,1), key=next(key))
        ]
        
    def __call__(self, x, key = None):
        out = x
        hiddens = [x]

        for activation, layer in zip(self.activations, self.layers):
            hiddens = [*hiddens, activation(layer(out))]
            out = jnp.concatenate(hiddens)

        out = hiddens[-1]

        return out * 0.2 + x

class RRDB(Module):
    net: Module

    def __init__(self, features, key = None):
        key = RNG(key)
        self.net = nn.Sequential([
            ResidualDenseBlock(features, features // 2, key = next(key)) 
            for _ in range(3)
        ])

    def __call__(self, x, key = None):
        out = self.net(x)
        out = out * 0.2 + x

        return out

class RRDBNet(Module):
    head: Module
    body: Module
    body_out: Module
    upscale: Module

    def __init__(self, channels:int, features:int, key = None):
        key = RNG(key)

        self.head = nn.Conv2d(channels, features, (3, 3), padding=(1,1), key=next(key))
        self.body = nn.Sequential([RRDB(features, key=next(key)) for _ in range(6)])
        self.body_out = nn.Conv2d(features, features, (3,3), padding=(1,1), key=next(key))

        self.upscale = nn.Sequential([
            Upscale(2), nn.Conv2d(features, features, (3,3), padding=(1,1), key=next(key)), LReLU(0.2),
            Upscale(2), nn.Conv2d(features, features, (3,3), padding=(1,1), key=next(key)), LReLU(0.2),
            nn.Conv2d(features, features, (3,3), padding=(1,1), key=next(key)), LReLU(0.2),
            nn.Conv2d(features, channels, (3,3), padding=(1,1), key=next(key))
        ])

    def __call__(self, x):
        head = self.head(x)
        body = self.body(head)
        body = self.body_out(body)

        x = head + body
        x = self.upscale(x)

        return x
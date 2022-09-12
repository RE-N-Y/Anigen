import jax
import jax.numpy as jnp
import jax.random as jr

import flax
import flax.linen as nn
from einops import rearrange

sg = jax.lax.stop_gradient

def normalise(x, axis=-1, eps=1e-12):
    return x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=axis, keepdims=True) + eps)

def euclidean_distance(samples, codes):
    distance = rearrange(samples, 'n d -> n () d') - rearrange(codes, 'c d -> () c d')
    distance = jnp.sum(distance ** 2, axis=-1)
    return distance

class VectorQuantiser(nn.Module):
    features:int = 768
    code_features:int = 16
    codes:int = 8192
    beta:float = 0.25

    def setup(self):
        self.codebook = nn.Embed(self.codes, self.code_features)
        self.input =  nn.Dense(self.code_features)
        self.output = nn.Dense(self.features)

    def __call__(self, z):
        z = self.input(z)
        z, codes = normalise(z), normalise(self.codebook.embedding)

        distance = euclidean_distance(z, codes)
        idxes = jnp.argmin(distance, axis=-1)
        codes = normalise(self.codebook(idxes))

        loss = self.beta * jnp.mean((sg(z) - codes) ** 2) + \
                           jnp.mean((z - sg(codes)) ** 2)

        codes = z + sg(codes - z)
        codes = self.output(codes)

        return codes, loss, idxes
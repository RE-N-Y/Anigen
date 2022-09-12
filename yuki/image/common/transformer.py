import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
from flax.linen.attention import make_causal_mask

import math
import numpy as onp
from einops import rearrange


pair = lambda t : (t,t)
zeros = jax.nn.initializers.constant(0)

class ViTEmbeddings(nn.Module):
    size:int
    patch:int
    features:int
    dropout:float

    @nn.compact
    def __call__(self, images):
        x = nn.Conv(self.features, pair(self.patch), pair(self.patch), padding = 0)(images)
        x = rearrange(x, 'b h w c -> b (h w) c')
        x += self.param("position_embeddings", zeros, (self.features, (self.size // self.patch) ** 2))
        x = nn.Dropout(self.dropout)(x)
        return x

class Attention(nn.Module):
    def setup(self, features:int, heads:int, length:int, dropout:float, autoregressive = False):
        self.features, self.heads = features, heads
        self.query, self.key, self.value = nn.Dense(features), nn.Dense(features), nn.Dense(features)
        self.mask = make_causal_mask(jnp.ones(length), dtype=bool) if autoregressive else None
        self.dropout = nn.Dropout(self.dropout)


    def __call__(self, x):
        B, T, D = x.shape
        q,k,v = self.query(x), self.key(x), self.value(x)
        q,k,v = map(lambda x : rearrange(x, 'b n (h d) -> b h n d', h = self.heads), (q,k,v))

        k = rearrange(k, 'b h n d -> b h d n')
        attention = q @ k / math.sqrt(self.features // self.heads)
        if self.mask is not None: attention = jnp.where(self.mask, attention[:,:T,:T], float('-inf'))

        attention = jax.nn.softmax(attention)
        attention = self.dropout(attention)

        outputs = rearrange(attention @ v, 'b h n d -> b n (h d)')
        outputs = self.dropout(outputs)

        return outputs

class Transformer(nn.Module):
    features:int = 768
    heads:int = 12
    dropout:float = 0

    def setup(self, features:int = 768, heads:int = 12, dropout:float = 0):
        self.attention = nn.Sequential([Attention(heads, features, dropout), nn.LayerNorm()])
        self.mlp = nn.Sequential([nn.Dropout(dropout), nn.Dense(features), nn.gelu, nn.Dense(features * 4), nn.LayerNorm()])

    def __call__(self, x):
        x = self.attention(x) + x
        x = self.mlp(x) + x
        return x
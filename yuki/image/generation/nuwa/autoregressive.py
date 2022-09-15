import jax
import jax.numpy as jnp
import jax.random as jr
import equinox
from equinox import nn, static_field, Module

import numpy as onp
from einops import rearrange
from yuki.helpers import RNG

lnormal = jax.nn.initializers.lecun_normal(in_axis = -1, out_axis = -2)

class NUWAttention(Module):
    query : jnp.ndarray
    key : jnp.ndarray
    value : jnp.ndarray
    out : jnp.ndarray
    dropout : nn.Dropout
    heads : int = static_field()
    head_features : int = static_field()
    features: int = static_field()

    def __init__(self, features:int, heads:int = 12, dropout:float = 0., key = None):
        key = RNG(key)

        self.heads = heads
        self.features = features
        self.head_features = features // heads

        self.query = lnormal(next(key), (features, features))
        self.key = lnormal(next(key), (features, features))
        self.value = lnormal(next(key), (features, features))
        self.out = lnormal(next(key), (features, features))
        self.dropout = nn.Dropout(dropout)


    def __call__(self, x, context, embeddings, key = None):
        akey, okey = jr.split(key)
        q,k,v = self.query.T @ x, self.key.T @ context + embeddings, self.value.T @ context
        
        q,k,v = map(lambda x : rearrange(x, '(h d) ... -> h ... d', h=self.heads), (q,k,v))
        
        scale = onp.sqrt(self.head_features)
        k = k = rearrange(k, 'h ... d -> h d ...')
        # q [ h M d ] @ k [h d (Nc M) ] = attention [h M (Nc M)]
        attention = q @ k / scale 
        attention = self.dropout(jax.nn.softmax(attention), key=akey)

        # attention [h M (Nc M)] @ v [h (Nc M) d] = [h M d] -> (h d) M
        outputs = rearrange(attention @ v, 'h n d -> (h d) n') 
        outputs = self.dropout(self.out(outputs), key=okey)

        return outputs
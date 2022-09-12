from collections import namedtuple
import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn

import math
import numpy as onp
from einops import rearrange

zeros = jax.nn.initializers.constant(0)

class TConfig:
    length:int = 1024
    heads:int = 12
    features:int = 768
    dropout:float = 0.
    depth:int = 12
    ntokens:int = 8192
    autoregressive = False

class Attention(nn.Module):
    config:TConfig

    def setup(self):
        self.query, self.config.key, self.config.value = nn.Dense(self.config.features), nn.Dense(self.config.features), nn.Dense(self.config.features)
        self.mask = jnp.tril(jnp.ones((1, self.config.length, self.config.length))) if self.config.autoregressive else None
        self.drop = nn.Dropout(self.config.dropout)

    def __call__(self, x, deterministic = False):
        q,k,v = self.query(x), self.key(x), self.value(x)
        q,k,v = map(lambda x : rearrange(x, 'b n (h d) -> b h n d', h = self.heads), (q,k,v))

        scale = math.sqrt(self.config.features // self.config.heads)
        k = rearrange(k, 'b h n d -> b h d n')
        attention = q @ k / scale
        
        if self.mask is not None:
            dinfo = jnp.finfo(attention.dtype)
            attention = jnp.where(self.mask == 1, attention, dinfo.min)
            
        attention = self.drop(jax.nn.softmax(attention))
        outputs = rearrange(attention @ v, 'b h n d -> b n (h d)')
        outputs = self.drop(outputs, deterministic = deterministic)

        return outputs


class TransformerBlock(nn.Module):
    config:TConfig

    def setup(self):
        self.prenorm = nn.LayerNorm()
        self.postnorm = nn.LayerNorm()
        self.attention = Attention()
        self.mlp = nn.Sequential([nn.Dense(4 * self.config.features), nn.gelu, nn.Dense(self.config.features)])
        self.drop = nn.Dropout(self.config.dropout)

    def __call__(self, x, deterministic = False):
        x = x + self.attention(self.prenorm(x))
        x = x + self.drop(self.mlp(self.postnorm(x)), deterministic = deterministic)

        return x

class Transformer(nn.Module):
    config:TConfig

    def setup(self):
        self.wte = nn.Embed(self.config.ntokens, self.config.features)
        self.wpe = self.param("wpe", zeros, (self.config.length, self.config.features))
        self.head = nn.Dense(self.config.ntokens, use_bias = False)
        self.drop = nn.Dropout(self.config.dropout)
        self.layernorm = nn.LayerNorm()
        self.layers = nn.Sequential([TransformerBlock() for _ in range(self.depth)])

    @nn.compact
    def __call__(self, tokens, deterministic = False):
        embeddings = self.drop(self.wte(tokens) + self.wpe, deterministic = deterministic)
        hiddens = self.layers(embeddings)
        hiddens = self.layernorm(hiddens)
        logits = self.head(hiddens)

        return logits

# import alpa
# import ray
# import optax
# from flax.training.train_state import TrainState

# ray.init()
# alpa.init(cluster = "ray")

# key = jr.PRNGKey(42)
# T = Transformer()
# tx = optax.adam(3e-4)

# params = T.init(key, jnp.ones((4, 1024)))
# state = TrainState.create(apply_fn=T.apply, params=params, tx=tx)


# method = alpa.PipeshardParallel(num_micro_batches = 2, layer_option = alpa.AutoLayerOption(2), stage_option = "auto")


# @alpa.parallelize(method=method)
# def step(state, batch):
#     def nll(params):
#         logits = jax.nn.softmax(state.apply_fn(params, batch))
#         labels = jax.nn.one_hot(batch, 8192)
#         loss = optax.softmax_cross_entropy(logits[:,:-1,:], labels[:,1:,:])
#         return loss.mean()

#     grads = alpa.grad(nll)(state.params)
#     new = state.apply_gradients(grads=grads)
#     return new


# step(state, jnp.ones((4, 1024)))
# alpa.shutdown()

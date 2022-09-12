import jax
import jax.numpy as jnp
import jax.random as jr

import flax
import flax.linen as nn
from einops import rearrange
from yuki.image.common import VectorQuantiser


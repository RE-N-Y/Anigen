import jax
import jax.numpy as jnp
import jax.random as jr

import flax
import flax.linen as nn
from einops import rearrange
from flaxmodels.stylegan2.discriminator import Discriminator
from yuki.image.common import VectorQuantiser, TransformerBlock, LPIPS

pair = lambda t : (t,t)
zeros = jax.nn.initializers.constant(0)
FTYPE = jnp.float32

key = jr.PRNGKey(42)
params = LPIPS.init(key, jnp.ones((4, 3, 256, 256)))

def lpips(x,y): return LPIPS.apply(params, x, y) 
def L2(x, axis=None): return jnp.sqrt(jnp.sum(x ** 2, axis=axis))


class VQConfig:
    size:int = 256
    patch:int = 8
    channels:int = 3
    features:int = 768
    code_features:int = 16
    codes:int = 8192
    beta:float = 0.25
    encoder_depth:int = 12
    decoder_depth:int = 12
    heads:int = 12
    dropout:float = 0.

class VQGAN(nn.Module):
    config:VQConfig

    def setup(self):
        self.length = (self.config.size // self.config.patch) ** 2
        self.projection = nn.Conv(self.config.features, pair(self.config.patch), strides = pair(self.config.patch)) # padding = 0
        self.ipe = self.param("ipe", zeros, (self.length, self.config.features))
        self.drop = nn.Dropout(self.config.dropout)
        self.codebook = VectorQuantiser(self.config.features, self.config.code_features, self.config.codes, self.config.beta)
        self.encoder = nn.Sequential(
            [TransformerBlock(self.config) for _ in range(self.config.encoder_depth)] + 
            [nn.LayerNorm(), nn.Dense(self.config.features), nn.tanh, nn.Dense(self.config.features)]
        )
        self.decoder = nn.Sequential(
            [TransformerBlock(self.config) for _ in range(self.config.decoder_depth)] + 
            [nn.LayerNorm(), nn.Dense(self.config.features), nn.tanh, nn.Dense(self.config.features)]
        )
        self.out = nn.Conv(self.config.channels * self.config.patch ** 2, (1,1)) # padding = 0

    def __call__(self, images, deterministic = False):
        embeddings = rearrange(self.projection(images), 'b h w c -> b (h w) c')
        hiddens = self.dropout(embeddings + self.ipe, deterministic = deterministic)
        hiddens = self.encoder(hiddens)
        hiddens, loss, idxes = self.codebook(hiddens)
        hiddens = self.decoder(hiddens + self.ipe)
        
        images = rearrange(hiddens, 'b (h w) c -> b h w c', h = self.config.size, w = self.config.size)
        images = self.out(images)

        return images, loss, idxes


@parallel
def DRstep(Dstate, batch, interval:int, gradient_penalty_weight = 1.0):
    @alpa.value_and_grad
    def gradient_penalty_loss(Dstate, images):
        y, pullback = jax.vjp(Dstate.apply_fn, Dstate.params, images)
        (gradients,) = pullback(jnp.ones(y.shape, dtype=FTYPE))
        penalty = L2(gradients) - 1
        loss = 0.5 * gradient_penalty_weight * interval * penalty ** 2

        return loss.mean(), { }

    (loss, metrics), grads = gradient_penalty_loss(Dstate, batch)
    new = Dstate.apply_gradients(grads=grads)

    return new, loss, metrics

@parallel
def Dstep(Gstate, Dstate, batch):
    @alpa.value_and_grad
    def discriminator_loss(Dstate, Gstate, x):
        fakes, loss, idxes = Gstate.apply_fn(x)
        fscores, rscores = Dstate.apply_fn(fakes), Dstate.apply_fn(x)
        loss = jax.nn.softplus(fscores) + jax.nn.softplus(-rscores)

        return loss.mean(), { "fake": fscores, "real": rscores }

    (loss, metrics), grads = discriminator_loss(Dstate, Gstate, batch)
    new = Dstate.apply_gradients(grads = grads)

    return new, loss, metrics

@parallel
def Gstep(
    Gstate, Dstate, images,
    weight = 1.0,
    perceptual_weight = 0.3,
    adversarial_weight = 0.3,
    l1_weight = 0.0,
    l2_weight = 1.0
):
    @value_and_grad
    def reconstruction_loss(Gstate, Dstate, x):
        r, loss, idxes = Gstate.apply_fn(x)
        l2, l1 = jnp.square(x - r), jnp.abs(x - r)
        perceptual = lpips(x,r)
        adversarial = jax.nn.softplus(-Dstate.apply_fn(r))

        total_loss = weight * loss.mean() + \
                     perceptual_weight * perceptual.mean() + \
                     adversarial_weight * adversarial.mean() + \
                     l1_weight * l1.mean() + \
                     l2_weight * l2.mean()

        return total_loss, { "loss": loss, "adversarial": adversarial, "l1": l1, "l2": l2, "perceptual": perceptual }

    (loss, metrics), grads = reconstruction_loss(Gstate, Dstate, images)
    new = Gstate.apply_gradients(grads = grads)

    return new, loss, metrics
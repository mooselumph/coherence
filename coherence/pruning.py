from typing import Tuple
from ..custom_types import Batch

import jax
import optax
import haiku as hk

import jax.numpy as jnp

from operator import mul

from ..train import update_params

from ..utils import ravel_pytree

def init_mask(params):
    def to_true(w):
        return jnp.full_like(w,True,dtype=bool)
    return jax.tree_map(to_true,params)


def apply_mask(params,mask):
    return jax.tree_map(mul,params,mask)


def masked_update(opt,loss_fn,mask):
    update_fn = update_params(opt,loss_fn)

    @jax.jit
    def update(
      params: hk.Params,
      opt_state: optax.OptState,
      batch: Batch,
    ) -> Tuple[hk.Params, optax.OptState]:
        new_params, opt_state = update_fn(params, opt_state, batch)
        new_params = apply_mask(new_params,mask)
        return new_params, opt_state

    return update

def threshold_prune(params,old_mask,fraction=0.1):

    # Get unmasked parameters
    p = ravel_pytree(params)
    ind = ravel_pytree(old_mask)
    p = p[ind]

    num_params = len(p)
    print(num_params)
    num_prune = jnp.ceil(fraction*num_params).astype(int)

    threshold = jnp.sort(jnp.abs(p))[num_prune]

    def to_mask(w):
        return jnp.abs(w) > threshold
    return jax.tree_map(to_mask,params)


def imp(key,train_fn,prune_fn,params,num_reps=10):

    mask = init_mask(params)

    masks = []
    branches = []

    for _ in range(num_reps):

        subkey, key = jax.random.split(key)

        # Train network
        params = train_fn(mask,subkey)
        branches.append(params)

        # Threshold weights
        mask = prune_fn(params,mask)
        masks.append(mask)

    return masks, branches

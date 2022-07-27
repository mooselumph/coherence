from typing import Tuple
from ..custom_types import Batch

import jax

import optax
import haiku as hk

from .pruning import init_mask, apply_mask

from ..train import update_params
from ..train_with_state import update_params as update_params_with_state

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

def masked_update_with_state(opt,loss_fn,mask):
    update_fn = update_params_with_state(opt,loss_fn)

    @jax.jit
    def update(
      params: hk.Params,
      state: hk.State,
      opt_state: optax.OptState,
      batch: Batch,
    ) -> Tuple[hk.Params, optax.OptState]:
        new_params, state, opt_state = update_fn(params, state, opt_state, batch)
        new_params = apply_mask(new_params,mask)
        return new_params, state, opt_state

    return update



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

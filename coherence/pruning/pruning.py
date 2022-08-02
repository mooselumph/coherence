import jax
import jax.numpy as jnp
import numpy as np

from ..utils import ravel_pytree

from . import get_unmasked_leaf, get_unmasked


def layerwise_threshold_prune(params,old_mask,plan):

    # Get unmasked parameters
    def get_mask(param, mask, plan_item):

        if plan_item == 1:
            return None

        p = get_unmasked_leaf(param,mask)

        num_params = len(p)
        num_prune = jnp.ceil(plan_item*num_params).astype(int)
        # threshold = np.partition(jnp.abs(p),num_prune)[num_prune]
        threshold = jnp.sort(jnp.abs(p))[num_prune]

        return jnp.flatnonzero(jnp.abs(param) > threshold)

    return jax.tree_map(get_mask,params, old_mask, plan)



def global_threshold_prune(params,old_mask,plan,fraction=0.1):

    extracted = get_unmasked(params,old_mask)
    p = ravel_pytree(extracted)

    num_params = len(p)
    num_prune = jnp.ceil(fraction*num_params).astype(int)
    
    # threshold = jnp.partition(jnp.abs(p),num_prune)[num_prune]
    threshold = jnp.sort(jnp.abs(p))[num_prune]

    def to_mask(param,old_mask,plan_item):
        if plan_item == 1:
            return None
        return jnp.flatnonzero(jnp.abs(param) > threshold)

    return jax.tree_map(to_mask,params,old_mask,plan)



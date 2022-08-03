
import jax
import jax.numpy as jnp
import numpy as np

from functools import partial


def init_mask(params):
    return jax.tree_map(lambda _: None,params)


def apply_mask_leaf(param,mask):

        if mask == None:
            return param

        t = param[mask]
        param = param*0
        param = param.at[mask].set(t)
        return param

def apply_mask(params,mask):
    return jax.tree_map(apply_mask_leaf,params,mask)


def get_unmasked_leaf(param, mask, flipped=False):

    # Not flipped
    if not flipped:
        if mask is None:
            return jnp.ravel(param)
        else:
            return param[mask]

    # Flipped
    if mask is None:
        return []
    else:
        bool_mask = np.ones_like(param,dtype=bool)
        bool_mask[mask] = False
        return param[bool_mask]



def get_unmasked(params, mask, flipped=False):
    return jax.tree_map(partial(get_unmasked_leaf,flipped=flipped),params,mask)
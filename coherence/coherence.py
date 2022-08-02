import jax
import jax.numpy as jnp
from numpy import ravel

from .utils import ravel_pytree

def ptwise_with_state(loss_fn):
    """
    Params:
        loss_fn: function(params, batch) --> loss 
    
    Returns:
        function(params, batch) --> pointwise gradients
    
    """
    
    def helper(params, state, batch):
        batch['image'] = jnp.expand_dims(batch['image'], axis=0)
        batch['label'] = jnp.expand_dims(batch['label'], axis=0)
        return jax.grad(loss_fn, has_aux=True)(params, state, batch)
    
    return jax.jit(jax.vmap(helper, in_axes=(None, None, 0), out_axes=0))


def ptwise(loss_fn):
    """
    Params:
        loss_fn: function(params, batch) --> loss 
    
    Returns:
        function(params, batch) --> pointwise gradients
    
    """
    
    def helper(params, batch):
        batch['image'] = jnp.expand_dims(batch['image'], axis=0)
        batch['label'] = jnp.expand_dims(batch['label'], axis=0)
        return jax.grad(loss_fn)(params, batch)
    
    return jax.jit(jax.vmap(helper, in_axes=(None, 0), out_axes=0))


def get_coherence(pt_grads):

    def helper(g):
        num_pts = g.shape[0]
        # return jnp.sum(g,axis=0)**2
        # return jnp.sqrt(jnp.sum(g,axis=0)**2 / jnp.sum(jnp.concatenate([g[i,:]**2 for i in range(num_pts)],axis=0),axis=0))
        return jnp.abs(jnp.sum(g,axis=0)) / jnp.sum(jnp.concatenate([jnp.abs(g[i,:]) for i in range(num_pts)],axis=0),axis=0)

    c = jax.tree_map(helper,pt_grads)

    return c


from .pruning import get_unmasked

def subnetwork_coherence(c,mask):

    extracted = get_unmasked(c,mask)
    c_flat = ravel_pytree(extracted)
    c_in = jnp.mean(c_flat)

    extracted = get_unmasked(c,mask,flipped=True)
    c_flat = ravel_pytree(extracted)
    c_out = jnp.mean(c_flat)

    return c_in, c_out

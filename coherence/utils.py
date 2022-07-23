import jax.numpy as jnp
import jax

def ravel_pytree_batched(pytree):
    leaves, treedef = jax.tree_flatten(pytree)
    batch_size = leaves[0].shape[0]
    return jnp.concatenate([jnp.reshape(elt,(batch_size,-1)) for elt in leaves],axis=1)

def ravel_pytree(pytree):
    leaves, treedef = jax.tree_flatten(pytree)
    return jnp.concatenate([jnp.ravel(elt) for elt in leaves],axis=0)
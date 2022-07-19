from typing import Tuple
from custom_types import Batch

import jax
import jax.numpy as jnp

import haiku as hk
import optax


def net_accuracy(net):  
    @jax.jit
    def accuracy(params: hk.Params, batch: Batch) -> jnp.ndarray:
        predictions = net.apply(params, batch["image"])
        return jnp.mean(jnp.argmax(predictions, axis=-1) == batch["label"])
    return accuracy


def update_params(opt,loss_fn):   
    @jax.jit
    def update(
      params: hk.Params,
      opt_state: optax.OptState,
      batch: Batch,
    ) -> Tuple[hk.Params, optax.OptState]:
        grads = jax.grad(loss_fn)(params, batch)
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state
    return update

# We maintain avg_params, the exponential moving average of the "live" params.
# avg_params is used only for evaluation (cf. https://doi.org/10.1137/0330046)
@jax.jit
def ema_update(params, avg_params):
    return optax.incremental_update(params, avg_params, step_size=0.001)

def do_training(
    update_fn, 
    accuracy_fn,
    params, 
    opt_state, 
    train, 
    train_eval, 
    test_eval, 
    epochs=10001, 
    print_epoch=1000,
    aux_fn=None,
    aux_epoch=10,
    ):

    # avg_params = params
    
    # Train/eval loop.
    for step in range(epochs):

        if step % print_epoch == 0:
            # Periodically evaluate classification accuracy on train & test sets.
            train_accuracy = accuracy_fn(params, next(train_eval))
            test_accuracy = accuracy_fn(params, next(test_eval))
            train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
            print(f"[Step {step}] Train / Test accuracy: "
                f"{train_accuracy:.3f} / {test_accuracy:.3f}.")

        # Do SGD on a batch of training examples.
        batch = next(train)

        if aux_fn != None and step % aux_epoch == 0:
            aux_fn(params, batch)

        params, opt_state = update_fn(params, opt_state, batch)

        # avg_params = ema_update(params, avg_params)
    
    return params


def softmax_xent_loss(net):
    """
    Params:
        net: network function (with apply and init funcs)
    
    Returns:
        loss_fn: function(params, batch) --> loss
        
    Creates softmax cross entropy loss function for a given network
    """
    @jax.jit
    def loss(params, batch):
        logits = net.apply(params, batch["image"])
        labels = jax.nn.one_hot(batch["label"], 10)

        l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
        softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
        softmax_xent /= labels.shape[0]

        return softmax_xent + 1e-4 * l2_loss
    
    return loss 

# Returns both network and loss given net function so as to not mistakenly use incompatible functions
def network_and_loss(net_fn, rng=False):
    if not rng:
        net = hk.without_apply_rng(hk.transform(net_fn))
    else:
        net = hk.transform(net_fn)
        
    loss_fn = softmax_xent_loss(net)
    
    return net, loss_fn
    
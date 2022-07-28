import jax
import jax.numpy as jnp
import numpy as np

from ..utils import ravel_pytree


from enum import Enum
class MaskFlag(Enum):
    INIT = 1
    EXCLUDED = 2


def init_mask(params,plan=None):
    if plan is None:
        return jax.tree_map(lambda _: MaskFlag.INIT,params)

    return jax.tree_map(lambda _, plan: MaskFlag.EXCLUDED if plan == 1 else MaskFlag.INIT,params,plan)


def apply_mask(params,mask):

    def zero_mask_complement(param,mask):

        if type(mask) == MaskFlag and (mask == MaskFlag.EXCLUDED or mask == MaskFlag.INIT):
            return param

        t = param[mask]
        param = param*0
        param = param.at[mask].set(t)
        return param

    return jax.tree_map(zero_mask_complement,params,mask)


def get_leaf_addresses(tree):

    def helper(tree,prefix=''):
        if isinstance(tree,dict):
            d = dict()
            for key in tree.keys():
                d[key] = helper(tree[key],f"{prefix}/{key}")
            return d
        return prefix

    return helper(tree)

# rule = [condition,action]

import re
class Rule(object):
    def __init__(self, condition, action):
        self.condition = condition
        self.action = action

    def check(self,address):
        if self.condition == None:
            return True
        if re.search(self.condition,address):
            return True
        return False
    
    def apply(self,value):
        if callable(self.action):
            return self.action(value)
        else:
            return self.action


def create_plan(params,rules,default_value=0):

    addresses = get_leaf_addresses(params)

    def apply_rules(address):
        value = default_value
        for rule in rules:
            if rule.check(address):
                value = rule.apply(value)
        return value

    return jax.tree_map(apply_rules,addresses)


# Get unmasked parameters
def extract_masked_params(param, mask, flipped=False):

    if type(mask) == MaskFlag and mask == MaskFlag.EXCLUDED:
        return []

    if not flipped:
        if type(mask) == MaskFlag:
            assert(mask == MaskFlag.INIT)
            return jnp.ravel(param)
        else:
            return param[mask]

    if type(mask) == MaskFlag:
        assert(mask == MaskFlag.INIT)
        return []
    else:
        bool_mask = np.ones_like(param,dtype=bool)
        bool_mask[mask] = False
        return param[bool_mask]


def layerwise_threshold_prune(params,old_mask,plan):

    # Get unmasked parameters
    def get_mask(param, mask, fraction):

        if type(mask) == MaskFlag and mask == MaskFlag.EXCLUDED:
            return MaskFlag.EXCLUDED

        p = extract_masked_params(param,mask)

        if len(p) < 10:
            print(mask)

        num_params = len(p)
        num_prune = jnp.ceil(fraction*num_params).astype(int)
        # threshold = np.partition(jnp.abs(p),num_prune)[num_prune]
        threshold = jnp.sort(jnp.abs(p))[num_prune]

        return jnp.nonzero(jnp.abs(p) > threshold)

    return jax.tree_map(get_mask,params, old_mask, plan)



def global_threshold_prune(params,old_mask,plan,fraction=0.1):

    extracted = jax.tree_map(extract_masked_params,params,old_mask)
    p = ravel_pytree(extracted)

    num_params = len(p)
    print(num_params)
    num_prune = jnp.ceil(fraction*num_params).astype(int)

    # threshold = jnp.partition(jnp.abs(p),num_prune)[num_prune]
    threshold = jnp.sort(jnp.abs(p))[num_prune]

    def to_mask(param,old_mask):
        if type(plan) == MaskFlag and plan == MaskFlag.EXCLUDED:
            return MaskFlag.EXCLUDED
        return jnp.nonzero(jnp.abs(param) > threshold)

    return jax.tree_map(to_mask,params,old_mask)



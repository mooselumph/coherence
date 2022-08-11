import jax
import jax.numpy as jnp

from . import MaskFlag

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



def flag_condition(flag_value):
    def check(input):
        return input == flag_value
    return check


def get_where(plan,tree,cond=flag_condition(1)):
    
    def get_where_helper(plan,leaf):
        if not cond(plan):
            return None
        return leaf

    return jax.tree_map(get_where_helper,plan,tree)


def apply_where(plan,f,*trees,cond=flag_condition(1)):

    def apply_where_helper(plan_item,*leaves):
        if not cond(plan_item):
            return None
        return f(*leaves)

    return jax.tree_map(apply_where_helper,plan,*trees)

  
def tighten_mask(mask,plan,cond=flag_condition(0)):
    return jax.tree_map(lambda m, plan_item: jnp.array([]) if cond(plan_item) else m, mask, plan)


def loosen_mask(mask,plan,cond=flag_condition(1)):
    return jax.tree_map(lambda m, plan_item: MaskFlag.ALL if cond(plan_item) else m, mask, plan)

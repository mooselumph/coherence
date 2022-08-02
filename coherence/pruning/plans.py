import jax

from enum import Enum
class PlanFlag(Enum):
    EXCLUDED = 1


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


# def apply_included(f,plan,*trees):

#   def if_included(plan,*params):
#     if plan == PlanFlag.EXCLUDED:
#       return None
#     return f(*params)

#   return jax.tree_map(if_included,plan,*trees)

  
def tighten_mask(mask,plan):
    return jax.tree_map(lambda m, plan_item: [] if plan_item == 0 else m, mask, plan)


def loosen_mask(mask,plan):
    return jax.tree_map(lambda m, plan_item: None if plan_item == 1 else m, mask, plan)
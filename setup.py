#!/usr/bin/env python

from setuptools import setup

setup(
    name='coherence',
    version='0.0.1',
    packages=['coherence'],
    install_requires=[
        'tensorflow',
        'tensorflow-datasets',
        'jax[cpu]',
        'dm-haiku @ git+https://github.com/deepmind/dm-haiku#egg=dm-haiku',
        'optax',
        'matplotlib',
        'tqdm',
    ],
)
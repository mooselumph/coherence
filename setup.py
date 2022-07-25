#!/usr/bin/env python

from setuptools import setup

setup(
    name='coherence',
    version='0.0.1',
    packages=['coherence'],
    install_requires=[
        'tensorflow',
        'tesnforflow_datasets',
        'jax[cpu]',
        'git+https://github.com/deepmind/dm-haiku',
        'optax',
        'matplotlib',
        'tqdm',
    ],
)
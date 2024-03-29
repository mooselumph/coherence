from .custom_types import Batch
from typing import Generator

import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

import os

def load_dataset(dataset: str, split: str, is_training: bool, 
                 batch_size: int, data_dir: str = '~/tensorflow_datasets', 
                 format_fun=None, filter_fn=None) -> Generator[Batch, None, None]:

    data_dir = os.path.expanduser(data_dir)

    if dataset == "cifar10":
        ds = tfds.load("cifar10", split=split, data_dir=data_dir).repeat()
    elif dataset == "mnist":
        ds = tfds.load("mnist:3.*.*", split=split, data_dir=data_dir).repeat()
    else:
        raise Exception("Invalid Dataset")
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)

    if format_fun != None:
        ds = ds.map(format_fun)

    if filter_fn != None:
        ds = ds.filter(filter_fn)

    ds = ds.batch(batch_size)
    return iter(tfds.as_numpy(ds))

def get_data_by_class(dset="mnist",batch_size=1000,data_dir='~/tensorflow_datasets',format_fun=None,labels=range(10)):

    datasets = []
    for label in labels:
        ds = load_dataset(dset, "train", True, batch_size, data_dir, format_fun, filter_fn=lambda fd: fd['label'] == label)
        datasets.append(ds)

    return datasets

def get_data(dset="mnist", batch_size=1000,data_dir='~/tensorflow_datasets',format_fun=None):
    train = load_dataset(dset, "train", True, batch_size, data_dir, format_fun)
    train_eval = load_dataset(dset, "train", False, batch_size, data_dir, format_fun)
    test_eval = load_dataset(dset, "test", False, batch_size, data_dir, format_fun)
    return train, train_eval, test_eval

def sanitize(d):
    d = normalize(d)
    del d['id']
    return d

def normalize(d):
    image = tf.cast(d['image'], tf.float32)
    # Normalize the pixel values
    d['image'] = image / 255.0
    return d

def decimate(d):
    d = normalize(d)
    d['image'] = tf.image.resize(d['image'], (14,14))
    return d
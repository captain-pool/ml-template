import pickle
import tensorflow as tf
import config
import pandas as pd

configuration = config.Config()


def load_dataset(key):
  raise NotImplementedError
  strategy = tf.distribute.get_strategy()
  path = config.dataset[key].path
  dataset = None # Load Dataset here
  if configuration.dataset.shuffle:
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
  dataset = dataset.batch(configuration.dataset[key].batch_size)
  if configuration.dataset[key].repeat:
    if configuration.dataset[key].repeat < 0:
      dataset = dataset.repeat()
    else:
      dataset = dataset.repeat(configuration.dataset[key].repeat)

  return iter(strategy.experimental_distribute_dataset(dataset))


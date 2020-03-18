import tensorflow as tf

def rmse_loss_fn(predictions, targets):
  return tf.sqrt(tf.reduce_mean((predictions - targets)**2))

def diff_loss_fn(uncompressed, original):
  raise NotImplementedError

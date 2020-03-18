import functools
import inspect
import tensorflow as tf
import tensorflow_addons as tfa
import config
import losses

configuration = config.Config() #pylint: disable=no-value-for-parameter

_STRATEGIES = {
    "cpu": [
        functools.partial(tf.distribute.OneDeviceStrategy, "/cpu:0"),
       "/cpu:0"],
    "gpu": [
        functools.partial(tf.distribute.OneDeviceStrategy, "/gpu:0"),
        "/gpu:0"]
}

_LOSSES = {
  "root_mean_squared_error": losses.rmse_loss_fn,
  "diff_loss": losses.diff_loss_fn
}

_OPTIMIZERS = {
  config.snake_case(class_name): class_ \
  for class_name, class_ in vars(tf.optimizers).items() \
  if inspect.isclass(class_) and \
    issubclass(class_, tf.optimizers.Optimizer)
}

_OPTIMIZERS.update({
    config.snake_case(class_name): class_ \
    for class_name, class_ in vars(tfa.optimizers).items() \
    if inspect.isclass(class_) and \
       issubclass(class_, tf.optimizers.Optimizer)
})

def get_optimizer(lr=None, name=None):
    optimizer_name = name or configuration.training.optimizer.name
    configuration.training.optimizer.lr = lr or configuration.training.optimizer.lr
    if not configuration.training.optimizer.lr:
      raise ValueError("Learning Rate not set. Please set the inital Learning Rate")
    if not optimizer_name:
      raise ValueError("Optimizer name not set")
    optimizer_fn = _OPTIMIZERS.get(optimizer_name)
    assert callable(optimizer_fn), \
       "Couldn't Find an optimizer with name: %s" % optimizer_name
    return optimizer_fn(**configuration.training.optimizer)

def get_learning_rate(step):
  if configuration.training.learning_rate.step:
    if step >= configuration.training.learning_rate.step[0]:
      configuration.training.learning_rate.step.pop(0)
      return configuration.training.learning_rate.value.pop(0)

def save_checkpoint(checkpoint):
  checkpoint.save(file_prefix=configuration.training.checkpoint_folder)

def checkpoint_exists():
  checkpoint_prefix = configuration.training.checkpoint_folder
  return tf.io.gfile.exists(checkpoint_prefix)\
         and tf.train.latest_checkpoint(checkpoint_prefix)

def load_latest_checkpoint(checkpoint):
  return checkpoint.restore(
      tf.train.latest_checkpoint(
          configuration.training.checkpoint_folder))

def get_loss(loss_name=None):
  loss_name = loss_name or configuration.training.loss
  loss_fn = _LOSSES.get(loss_name)

  if not loss_fn:
    loss_fn = tf.keras.losses.get(loss_name)

  return loss_fn

def get_strategy(device):
  if "gpu" in device:
    assert tf.test.is_gpu_available(), "No GPUs Found."
  strategy, device = _STRATEGIES[device]
  strategy = strategy()
  return strategy, device

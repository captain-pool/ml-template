from __future__ import print_function

import datetime
import importlib
import os
import time

import argparse
import config
import tqdm
import tensorflow as tf
import tensorflow_addons as tfa

def build_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-d", "--device", default="cpu",
      help="Device to use for training")
  parser.add_argument(
      "--configs", dest="config", nargs="+", required=True,
      help="Config Files to load Configuration From")
  parser.add_argument(
      "--type", dest="model.type", type=str,
      help="Type of Model to train on" \
      "Can take only (vanilla/ residual)")
  parser.add_argument(
      "--batch_size", dest="dataset.training.batch_size",
      type=int, help="Batch Size to Work on")
  parser.add_argument(
      "--train_path", dest="dataset.training.path",
      type=os.path.normpath, help="Path to traning data source")
  parser.add_argument(
      "--test_path", dest="dataset.testing.path",
      type=os.path.normpath, help="Path to testing data source")
  parser.add_argument(
      "--num_steps", dest="training.num_steps",
      type=int, help="Number of steps to train for")
  parser.add_argument(
      "--logdir", dest="training.logdir", type=os.path.normpath,
      help="Path to Log Directory")
  parser.add_argument(
      "--export_config", default=False, action="store_true",
      help="exports the complete config")
  parser.add_argument(
      "--export_saved_model", default=None,
      dest="training.export_saved_model_dir",
      help="Path to export Saved Model to")
  return parser

def train(model, dataset, checkpoint_step,
          train_summary_writer, test_summary_writer,
          batch_size, num_steps,
          test_dataset, test_step, test_num_steps):
  loss_fn = utils.get_loss()
  step = 0
  progress_bar = tqdm.tqdm(total=num_steps)
  tf.summary.experimental.set_step(tf.Variable(0, tf.int64))
  learning_rate = utils.get_learning_rate(0)
  optimizer = utils.get_optimizer(lr=learning_rate)
  train_cumulative_loss = tf.metrics.Mean()
  test_cumulative_loss = tf.metrics.Mean()

  loss_scale_factor = max(
      1, batch_size // tf.distribute.get_strategy().num_replicas_in_sync)
  checkpoint = tf.train.Checkpoint(
      optimizer=optimizer,
      model=model,
      summary_step=tf.summary.experimental.get_step())
  if utils.checkpoint_exists():
    utils.load_latest_checkpoint(checkpoint)

  def _train_step_fn(model, X, Y):
    with tf.GradientTape() as tape:
      output = model(X)
      loss = loss_fn(output, Y) * (1 / batch_size) * loss_scale_factor
      train_cumulative_loss(loss)
    gradient = tape.gradient(loss, model.trainable_variables)
    step_op = optimizer.apply_gradients(
        zip(gradient, model.trainable_variables))
    tf.summary.experimental.set_step(optimizer.iterations)
    with tf.control_dependencies([step_op]):
      return tf.summary.experimental.get_step()

  def _test_step_fn(model, X, Y):
    output = model(X)
    reconstruction_loss = loss_fn(output, Y) * (1 / batch_size) * loss_scale_factor
    test_cumulative_loss(reconstruction_loss)
    add_op = tf.summary.experimental.get_step().assign_add(1)
    with tf.control_dependencies([add_op]):
      return tf.summary.experimental.get_step()

  @tf.function
  def distributed_step(model, X, Y, istrain):
    strategy = tf.distribute.get_strategy()
    _step_fn = _test_step_fn
    if istrain:
      _step_fn = _train_step_fn

    distributed_step, distributed_diff = strategy.experimental_run_v2(
        _step_fn, args=[model, X, Y])
    mean_step = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                  distributed_step, axis=None)
    return mean_step

  with train_summary_writer.as_default():
    while step < num_steps:
      X, Y = next(dataset)
      start_time = time.time()
      step, diff = distributed_step(model, X, Y, True)
      end_time = time.time()
      learning_rate = utils.get_learning_rate(step)
      if learning_rate:
        optimizer.learning_rate.assign(learning_rate)
      if not step % checkpoint_step:
        tf.print("Checkpoining ...")
        utils.save_checkpoint(checkpoint)

      tf.summary.scalar("Cumulative Reconstruction Loss",
                        train_cumulative_loss.result(),
                        tf.summary.experimental.get_step())
      tf.summary.scalar("Train Step Time (in seconds)",
                        end_time - start_time,
                        tf.summary.experimental.get_step())


      if test_dataset is not None:
        with test_summary_writer.as_default():
          while (step % test_step) < test_num_steps:
            test_X, test_Y = next(test_dataset)
            step, diff = distributed_step(model, test_X, test_Y, False)
            utils.plot_diff_loss(diff)
            progress_bar.update(1)
            tf.summary.scalar("Cumulative Reconstruction Loss",
                              test_cumulative_loss.result(),
                              tf.summary.experimental.get_step())

      progress_bar.update(1)
  progress_bar.close()

def main(argv):
  tf.random.set_seed(0)
  print("Using Configuration")
  print(argv)
  strategy, device = utils.get_strategy(argv.device)
  timestamp = datetime.datetime.now().isoformat()
  with tf.device(device), strategy.scope():
    train_data = dataset.load_dataset("training")
    test_data = None if not argv.dataset.testing.path else \
                dataset.load_dataset("testing")
    model = models.load_model(argv.models.type)
    if not tf.io.gfile.exists(argv.training.logdir):
      tf.io.gfile.makedirs(argv.training.logdir)
    if not tf.io.gfile.exists(
        os.path.dirname(argv.training.checkpoint_folder)):
      tf.io.gfile.makedirs(
          os.path.dirname(argv.training.checkpoint_folder))

    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(argv.training.logdir, "expt_%s" % timestamp, "train"))
    test_summary_writer = tf.summary.create_file_writer(
        os.path.join(argv.training.logdir, "expt_%s" % timestamp, "test"))
    train(model, train_data, argv.training.checkpoint_step,
          train_summary_writer, test_summary_writer,
          argv.dataset.training.batch_size, argv.training.num_steps,
          test_data, argv.testing.step, argv.testing.step_size)

    if argv.training.export_saved_model_dir:
      print("Saving Model to %s" % argv.training.export_saved_model_dir)
      tf.saved_model.save(model, argv.training.export_saved_model_dir)


if __name__ == "__main__":
  parser = build_parser()
  args = parser.parse_args()
  configs = config.Config(args)
  # Importing Later on since these modules
  # needs configuration objects to already be
  # initialized with the arguments
  dataset_module = importlib.import_module("dataset")
  models_module = importlib.import_module("models")
  utils_module = importlib.import_module("utils")
  globals().update({
      "dataset": dataset_module,
      "models": models_module,
      "utils": utils_module
  })
  main(configs)

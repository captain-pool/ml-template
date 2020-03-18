import tensorflow as tf
import re
import config

class ModelRegister(type):
  instances = {}
  def __init__(cls, *args, **kwargs):
    class_name = config.snake_case(cls.__name__).split("_")[0]
    if class_name not in ModelRegister.instances and \
      class_name != "model":
      ModelRegister.instances[class_name] = cls

class Layer(tf.keras.layers.Layer, metaclass=ModelRegister):
  def __init__(self, *args, **kwargs):
    super(Layer, self).__init__(*args, **kwargs)

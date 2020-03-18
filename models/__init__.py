from models.abstract import ModelRegister
import config
import tensorflow as tf

configuration = config.Config().models

def load_model(name):
  raise NotImplementedError

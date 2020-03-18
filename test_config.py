import argparse
import glob
import config

namespace = argparse.Namespace(config=glob.glob("configs/*yaml"))
cfg = config.Config(namespace)

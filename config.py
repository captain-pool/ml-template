import re
import addict
import confuse

def singleton(cls):
  instances = {}
  def getinstance(*args, **kwargs):
    if kwargs.get("orphan"):
      kwargs.pop("orphan")
      return cls(*args, **kwargs)

    if cls not in instances:
      instances[cls] = cls(*args, **kwargs)
    return instances[cls]
  return getinstance

def snake_case(string):
  string= re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
  return re.sub('([a-z0-9])([A-Z])', r'\1_\2', string).lower()

@singleton
class Config(addict.Dict):
  def __init__(self, args):
    _parsed = self._parse(args, args.config)
    second_copy = addict.Dict(_parsed)
    self.update(second_copy)


  def _parse(self, args, configs):
    config = confuse.Configuration("Config")
    for config_files in configs:
      config.set_file(config_files)
    config.set_args(args, dots=True)
    flattened = config.flatten()
    export_config_path = flattened.get("export_config_path")
    export_config = bool(flattened.get("export_config"))
    if export_config_path and export_config:
      with open(export_config_path, "w") as f:
        f.write(config.dump())
    return flattened

import functools
import inspect
from typing import Dict

import yaml

_global_config = None


def get(key: str):
    if _global_config is None:
        raise Exception('no global config defined')
    return _global_config.resolve_param(key)


def configure(func):
    @functools.wraps(func)
    def wrapper(*kargs, **kwargs):
        # if no config is available call the function as is
        if _global_config is None:
            return func(*kargs, **kwargs)

        sig = inspect.signature(func)
        for first_arg_name, first_arg in inspect.signature(func).parameters.items():
            break
        else:
            raise Exception('do not add configure decorator to lists without arguments')

        cls = kargs[0].__class__ if first_arg_name == 'self' else None

        params = {}
        for i, (param_name, param) in enumerate(sig.parameters.items()):
            if i == 0 and cls is not None:
                continue
            try:
                #  TODO verify parameter type
                params[param_name] = _global_config.resolve_param(param_name, cls)
            except KeyError:
                pass

        f = functools.partial(func, *kargs, **params)
        try:
            result = f(**kwargs)
        except TypeError as e:
            if cls:
                raise Exception('configuration error calling creating object {:s}:'.format(cls.__name__), e) from None
            else:
                raise Exception('missing configuration value?', e) from None
        return result

    return wrapper


class Config:
    def __init__(self, config=None, config_file=None, make_global=True, loggable_params=None):
        self.loggable_params = loggable_params
        if config is None and config_file is None:
            raise ValueError("either config or config_file needs to be specified")

        def merge_config(acc, v):
            acc.update(v)
            return acc

        if config is not None:
            self._config = config
        else:
            self._config = {}

        if config_file is not None:
            with open(config_file, 'r') as f:
                cfg = yaml.safe_load_all(f)
                self._config.update(functools.reduce(merge_config, cfg))

        if make_global:
            self.make_global()

    def set_param(self, parameter_name, value, requesting_class=None):
        if requesting_class is not None:
            self._config[requesting_class.__name__][parameter_name] = value
        else:
            self._config[parameter_name] = value
        return self

    def resolve_param(self, parameter_name: str, requesting_class=None):
        if requesting_class is None:
            return self._config[parameter_name]

        mro = inspect.getmro(requesting_class)
        for cls in mro:
            if cls.__name__ in self._config and parameter_name in self._config[cls.__name__]:
                return self._config[cls.__name__][parameter_name]

        # check for default value
        if parameter_name in self._config:
            return self._config[parameter_name]

        if requesting_class is not None:
            raise KeyError(
                'parameter {} not configured for class {}'.format(parameter_name, requesting_class.__name__))
        raise KeyError('no global parameter {} configured'.format(parameter_name))

    def make_global(self):
        global _global_config
        _global_config = self
        return self

    def get_params(self) -> Dict[str, any]:
        return self._config


def get_global_config() -> Config:
    if _global_config is None:
        raise Exception('no global config defined')
    return _global_config


def get_loggable_params():
    params = get_global_config().get_params()

    def _flatten_and_filter(dic):
        result = {}
        for k, v in dic.items():
            if not any((isinstance(v, t) for t in (str, int, float, dict, list))):
                continue

            if not isinstance(k, str):
                raise Exception('only params with string keys can be logged')
            if isinstance(v, dict):
                result.update(_flatten_and_filter(v))
            else:
                result[k] = v
        return result

    return _flatten_and_filter(params)
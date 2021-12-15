# import json
from pathlib import Path
from pydantic import BaseModel, Extra, Json, root_validator
# from pydantic import Protocol, StrBytes
import traceback
from typing import Any, Type, Union
# import ujson
import yaml


def check_unique(keys: list, err_msg='Duplicate item(s): {}') -> None:
    if len(set(keys)) < len(keys):
        s_, errs = set(), []
        [errs.append(f'"{x}"') if x in s_ else s_.add(x) for x in keys]
        errs = ', '.join(errs)
        errs = err_msg.format(errs)
        raise ValueError(errs)

# YAML_TAG = '<yaml>'
# def obj_load(data):
#     if data.startswith(YAML_TAG):
#         res = yaml.safe_load(data[len(YAML_TAG):])
#     else:
#         json.loads(data, object_pairs_hook=\
#                              lambda x: check_unique(list(x[0] for x in x)))
#         res = ujson.loads(data)
#     return res

def no_duplicates_constructor(loader, node, deep=False):
    check_unique(list(loader.construct_object(x[0], deep=deep)
                          for x in node.value))
    return loader.construct_mapping(node, deep)

yaml.SafeLoader.add_constructor(yaml.resolver.BaseResolver
                                             .DEFAULT_MAPPING_TAG,
                                no_duplicates_constructor)

class _BaseModel(BaseModel, extra=Extra.forbid, json_loads=yaml.safe_load):

    @classmethod
    def parse(cls: Type['Model'], config: Any) -> 'Model':
        is_path = isinstance(config, str) or isinstance(config, Path)
        if is_path:
            cls.config_path = config  # NB
        try:
            return cls.parse_obj(config) if isinstance(config, Json) else \
                   cls.parse_file(config) if is_path else \
                   cls.parse_raw(config)
        except RuntimeError as e:
            raise e
        except Exception as e:
            if is_path:
                err = traceback.format_exc()
                head = f'Error occurred while parsing "{config}"'
                raise RuntimeError(head + '\n' + '-' * len(head)  + '\n'
                                 + err)
            else:
                raise e

    @root_validator
    def _root_validator(cls, values):
        if hasattr(cls, 'config_path'):
            values['config_path'] = cls.config_path
            del cls.config_path
        return values

# class _BaseModel(BaseModel, extra=Extra.forbid, json_loads=obj_load):

#     @classmethod
#     def parse_raw(cls: Type['Model'], b: StrBytes, *,
#                   content_type: str = None, proto: Protocol = None,
#                   **kwargs) -> 'Model':
#         if proto == 'yaml' or (proto is None
#                            and content_type
#                            and content_type.endswith('yaml')):
#             b = (YAML_TAG if isinstance(b, str) else YAML_TAG.encode()) + b
#             proto = Protocol.json
#         return super().parse_raw(b, content_type=content_type, proto=proto,
#                                  **kwargs)

#     @classmethod
#     def parse_file(cls: Type['Model'], path: Union[str, Path], *,
#                   content_type: str = None, proto: Protocol = None,
#                   **kwargs) -> 'Model':
#         path = Path(path)
#         b = path.read_bytes()
#         if proto is None and content_type is None:
#             if path.suffix in ('.js', '.json'):
#                 proto = Protocol.json
#             elif path.suffix == '.yaml':
#                 proto = 'yaml'
#             elif path.suffix in ('.pkl', '.pickle'):
#                 proto = Protocol.pickle
#         return cls.parse_raw(b, content_type=content_type, proto=proto,
#                              **kwargs)

#     class Config:
#         extra = Extra.forbid
#         json_loads = json_loads

from enum import Enum
import os
from pydantic import DirectoryPath, FilePath, root_validator
from typing import Optional, Union

from .base_schema import _BaseModel


class FormatEnum(str, Enum):
    torch = 'torch'


class Model(_BaseModel):
    directory: DirectoryPath
    format: FormatEnum
    name: Union[DirectoryPath, FilePath]
    device: str

    swagger_title: str
    swagger_version: str
    swagger_description: Optional[str] = None

    @root_validator(pre=True)
    def _root_validator(cls, values):
        values['name'] = \
            os.path.join(values.get('directory', ''), \
                         values.get('name', ''))
        return values

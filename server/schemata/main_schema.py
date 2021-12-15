from enum import Enum
from ipaddress import IPv4Address
import os
from pydantic import DirectoryPath, FilePath, root_validator, validator
from typing import Dict, List, Union

from .base_schema import _BaseModel, check_unique
from .user_schema import Users


config = {}


class FormatEnum(str, Enum):
    torch = 'torch'


class Config(_BaseModel):
    host: IPv4Address
    port: int

    users: Union[FilePath, Dict]

    model_directory: DirectoryPath
    model_format: FormatEnum
    model_name: Union[DirectoryPath, FilePath]
    model_device: str

    password_hash_schemes: Union[str, List[str]]
    jwt_secret_key: str
    jwt_algorithm: str
    jwt_expire_minutes: int

    @validator('password_hash_schemes')
    def _validator_password_hash_schemes(cls, value):
        return value if isinstance(value, list) else [value]

    @validator('users')
    def _validator_users(cls, value):
        return Users.parse(value).__root__

    @root_validator(pre=True)
    def _root_validator(cls, values):
        values['model_name'] = \
            os.path.join(values.get('model_directory', ''), \
                         values.get('model_name', ''))
        return values

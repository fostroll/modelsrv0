from pydantic import BaseModel
from typing import Dict, List, Optional, Union

from .base_schema import _BaseModel


class UserData(_BaseModel):
    password: str
    full_name: str
    admin: bool = False
    scopes: Optional[Union[str, List[str]]] = None
    disabled: bool = False

class Users(_BaseModel):
    __root__: Dict[str, UserData]

class UserDataView(BaseModel):
    full_name: str
    admin: bool
    scopes: Union[str, List[str]]
    disabled: bool

from pydantic import BaseModel, Json, validator
from typing import Dict, List, Optional, Union


class Intents(BaseModel):
    __root__: Union[Dict[int, str], Dict[str, int], List[str]]

    @validator('__root__')
    def _validator_root(cls, value):
        if isinstance(value, List):
            value = {i: x for i, x in enumerate(value)}
        elif value and isinstance(value[0], str):
            value = {y: x for x, y in value.items()}
        return value

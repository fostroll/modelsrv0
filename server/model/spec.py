import json
import os
from pydantic import BaseModel, validator
from threading import Lock
import torch
from toxine import TextPreprocessor
from typing import Dict, List, Union


model = None
intents = None
intents_fn = None
text_preprocessor = TextPreprocessor()
lock = Lock()


class Intents(BaseModel):
    __root__: Union[List[str], Dict[str, int], Dict[int, str]]

    @validator('__root__')
    def _validator_root(cls, value):
        if isinstance(value, Dict):
            if value:
                value_ = {y: x for x, y in value.items()} \
                             if isinstance(next(iter(value)), str) else \
                         value
                value = []
                for idx, (intent_no, intent) in enumerate(
                    sorted(value_.items())
                ):
                    assert idx == intent_no, \
                       f'ERROR: Intent {idx} is missing in file "{intents_fn}".'
                    value.append(intent)
            else:
                value = []
        return value

from .model import Model


def model_load(path, device='cpu'):
    global model, intents, intents_fn
    model = Model.load(path)
    model.to(device)
    intents_fn = os.path.join(path, 'intents.json')
    intents = Intents.parse_file(intents_fn).__root__

def model_predict(text, with_intents=True, probs=True, threshold=.5,
                  only_true=False):
    assert model, 'ERROR: Invoke model_load() first.'
    with lock:
        text = text_preprocessor.process_text(text, silent=True)
    text = [x['FORM'] for x in text for x in x[0]]
    with lock, torch.no_grad():
        preds = model([text], [None], [None])[0]
    if not probs:
        preds = (preds >= threshold).bool()
    preds = preds.tolist()
    if with_intents:
        preds = {x: y for x, y in zip(intents, preds)}
        if only_true:
            preds = {x: y for x, y in preds.items()
                          if y is True or y >= threshold}
    elif only_true:
        preds = [x for x in preds if x is True or x >= threshold]
    return preds

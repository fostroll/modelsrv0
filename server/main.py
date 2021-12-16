#!python
#-*- coding: utf-8 -*-
import argparse
from fastapi import BackgroundTasks, Body, Depends, FastAPI, HTTPException, \
                    Response, Security, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import gc
import traceback
from typing import Dict, Tuple
import uvicorn

from auth import make_routes as auth_make_routes
from const import CONFIG_FN, ADMIN_PATH_PREFIX, PREDICT_PATH_PREFIX, \
                  STATIC_PATH, ADMIN_TAG, MODEL_TAG, LOADING_LOG
from model import model_load, model_predict
import response_examples
import schemata
from threading import Lock


new_config_lock, reloading_lock = Lock(), Lock()

app = FastAPI()

def load_config():
    schemata.config = schemata.Config.parse(CONFIG_FN)

def reload_model():
    with reloading_lock,\
         open(LOADING_LOG, 'wt', encoding='utf-8') as f:
        try:
            model_load(schemata.config.model.name,
                       device=schemata.config.model.device)
        except Exception as e:
            print(traceback.format_exc())
            print(traceback.format_exc(), file=f)
        else:
            print('The model is loaded.', file=f)

def load_router():

    app_ = FastAPI(title=schemata.config.model.swagger_title,
                   version=schemata.config.model.swagger_version,
                   description=schemata.config.model.swagger_description,
                   responses={**response_examples.HTTP_400_BAD_REQUEST,
                              **response_examples.HTTP_401_UNAUTHORIZED})
    if STATIC_PATH:
        app_.mount(STATIC_PATH,
                   StaticFiles(directory='static'), name='static')

    check_user = auth_make_routes(app_)

    def check_admin(current_user: schemata.UserData = Security(check_user)):
        if not current_user.admin:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail='Admin rights required')
        return current_user

    @app_.get(f'{ADMIN_PATH_PREFIX}/reload',
              name='Reload the model',
              #status_code=status.HTTP_204_NO_CONTENT)
              responses={**response_examples.HTTP_202_ACCEPTED,
                         **response_examples.HTTP_500_INTERNAL_SERVER_ERROR,
                         **response_examples.HTTP_503_SERVICE_UNAVAILABLE},
              dependencies=[Depends(check_admin)],
              tags=[ADMIN_TAG])
    async def admin_reload(background_tasks: BackgroundTasks):
        with new_config_lock:
            if reloading_lock.locked():
                raise HTTPException(status_code=\
                                        status.HTTP_503_SERVICE_UNAVAILABLE,
                                    detail='Process is locked. Another '
                                           'reloading is still in progress')
            with reloading_lock:
                try:
                    load_config()
                except RuntimeError as e:
                #except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(e).split('\n')
                    )
                load_router()
                background_tasks.add_task(reload_model)
                return JSONResponse('The request is processing',
                                    status_code=status.HTTP_202_ACCEPTED)

    @app_.get(f'{ADMIN_PATH_PREFIX}/reload/status',
              name='the model reloading status',
              responses={**response_examples.HTTP_503_SERVICE_UNAVAILABLE},
              dependencies=[Depends(check_admin)],
              tags=[ADMIN_TAG])
    async def admin_reload_check(t: int):
        if reloading_lock.locked():
            raise HTTPException(status_code=\
                                    status.HTTP_503_SERVICE_UNAVAILABLE,
                                detail='Reloading is still in progress')
        return FileResponse(LOADING_LOG)

    @app_.post('/predict',
               dependencies=[Security(check_user)],
               tags=[MODEL_TAG])
    async def predict(text: str = Body(...),
                      with_intents: bool = True, probs: bool = True,
                      threshold: float = .5, only_true: bool = False):
        return model_predict(text, with_intents=with_intents, probs=probs,
                             threshold=threshold, only_true=only_true)

    for attr, val in app_.__dict__.items():
        setattr(app, attr, val)

load_config()
load_router()
reload_model()

if __name__ == '__main__':
    #https://www.uvicorn.org/settings/
    #uvicorn.run(app, host='127.0.0.1', port=8000)

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    argparser = argparse.ArgumentParser(add_help=True)
    argparser.add_argument('--host', dest='host', type=str,
                           default='127.0.0.1', help='The server address')
    argparser.add_argument('--port', dest='port', type=int,
                           default=8000, help='The server port')
    argparser.add_argument('--reload', dest='reload', type=str2bool,
                           default=True, help='Whethere we need a reload')
    argparser.add_argument('--workers', dest='workers', type=int,
                           default=1, help='The number of workers')
    args = argparser.parse_args()
    uvicorn.run('main:app', host=args.host, port=args.port,
                reload=args.reload, workers=args.workers)

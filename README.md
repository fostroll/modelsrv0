# modelsrv0

The simple FastAPI wrapper for ML model.

## Requirements

```sh
pip install python-multipart
pip install fastapi
pip install uvicorn
pip install passlib
pip install python-jose
```

## Usage

1. Edit `config.yaml`. At least, you wanna change `jwt_secret_key` value with
```sh
openssl rand -hex 512
```

2. Edit user list in `config_users.yaml`. The hash for `password` key may be
generated with
```sh
python make_hash.py <password>
```

3. Replace script `server/model/spec.py` with your own, that is implemented
methods `model_load(path, device='cpu')` and `model_predict(text,
with_intents=True, probs=True, threshold=.5, only_true=False)`. Scripts
`server/model/model*.py` may be removed or replaced with your own model
definition.

4. Put the model in the directory `model0` (or change the `model_name` in
`config.yaml`).

5. Edit `run.sh` or `run.bat` and start the server.

## License

***modelsrv0*** is released under the Creative Commons License. See the
[LICENSE](https://github.com/fostroll/srv_zero/blob/master/LICENSE) file for
more details.

# /bin/sh

#uvicorn main:app --reload --app-dir server
python server/main.py --host=127.0.0.1 --port=8000 --reload=yes --workers=1

from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import OAuth2PasswordBearer, \
                             OAuth2PasswordRequestForm, SecurityScopes
from ipaddress import IPv4Address
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, ValidationError
from typing import List, Optional

import response_examples
import schemata
from schemata import UserData, UserDataView


LOGIN_PATH = '/login'


class Token(BaseModel):
    access_token: str
    token_type: str

def make_routes(router, login_path=LOGIN_PATH):
    pwd_context = CryptContext(schemes=schemata.config.password_hash_schemes,
                               deprecated='auto')

    oauth2_scheme = OAuth2PasswordBearer(
        tokenUrl='login',
        #scopes={'admin': 'Supervisor access only'}
    )

    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(password: str) -> str:
        return pwd_context.hash(password)

    def get_user(username: str) -> Optional[str]:
        return schemata.config.users.get(username)

    def authenticate_user(username: str, password: str) -> bool:
        user = get_user(username)
        return bool(user and verify_password(password, user.password))

    def create_access_token(data: dict,
                            expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({'exp': expire})
        encoded_jwt = jwt.encode(to_encode, schemata.config.jwt_secret_key,
                                 algorithm=schemata.config.jwt_algorithm)
        return encoded_jwt

    async def get_current_user(
        security_scopes: SecurityScopes, request: Request,
        token: str = Depends(oauth2_scheme)
    ) -> UserData:
        if security_scopes.scopes:
            authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
        else:
            authenticate_value = f'Bearer'
        err = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail='Could not validate credential',
                            headers={'WWW-Authenticate': authenticate_value})
        try:
            payload = jwt.decode(token, schemata.config.jwt_secret_key,
                                 algorithms=[schemata.config.jwt_algorithm],
                                 audience=request.client.host,
                                 options={'require_aud': True,
                                          'require_sub': True})
            username: str = payload['sub']
            token_scopes = payload.get('scopes', [])
        except (KeyError, JWTError, ValidationError):
            raise err
        user: UserData = get_user(username=username)
        if user is None:
            raise err
        for scope in security_scopes.scopes:
            if scope not in token_scopes:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail='Not enough permissions',
                    headers={'WWW-Authenticate': authenticate_value},
                )
        return user

    async def check_user(
        current_user: UserData = Security(get_current_user, scopes=[])
    ):
        if current_user.disabled:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail='Inactive user')
        return current_user

    @router.post(login_path, response_model=Token)
    async def login(request: Request,
                    form_data: OAuth2PasswordRequestForm = Depends()):
        if not authenticate_user(form_data.username, form_data.password):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail='Incorrect username or password')
        for scope in form_data.scopes:
            if scope not in schemata.config.users[form_data.username].scopes:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                    detail='Wrong scope set')
        access_token_expires = \
            timedelta(minutes=schemata.config.jwt_expire_minutes)
        access_token = create_access_token(
            data={'sub': form_data.username, 'aud': [request.client.host],
                  'scopes': form_data.scopes},
            expires_delta=access_token_expires,
        )
        return {'access_token': access_token, 'token_type': 'bearer'}

    @router.get('/users/me/', response_model=UserDataView)
    async def users_me(current_user: UserData = Security(check_user)):
        return current_user

    return check_user

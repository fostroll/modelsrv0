#!/usr/bin/python
from passlib.context import CryptContext
import sys


pwd_context = CryptContext(schemes='sha512_crypt', deprecated='auto')
print(pwd_context.hash(sys.argv[1]))

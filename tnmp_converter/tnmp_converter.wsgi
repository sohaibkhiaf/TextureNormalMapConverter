import sys

sys.path.insert(
    0,
    "/var/www/tnmp_converter"
)

from app import app

application = app


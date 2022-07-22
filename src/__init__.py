import logging

from src.config import get_config

FORMAT = '[%(asctime)s | %(levelname)s]\t%(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

config = get_config()
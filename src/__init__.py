import logging

from src.config import get_config, get_directories


FORMAT = '[%(asctime)s | %(levelname)s]\t%(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

config = get_config()
directories = get_directories(config)

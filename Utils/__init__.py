import logging
import os


import logging
import os

os.makedirs('Logs', exist_ok=True)

formatter = logging.Formatter('[%(levelname)s] - %(message)s')

file_handler = logging.FileHandler('Logs/logfile.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler], force=True)

def get_logger() -> logging.Logger:
    return logging.getLogger(__name__)


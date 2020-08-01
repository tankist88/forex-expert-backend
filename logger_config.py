import sys
import logging
import logging.config
from logging.handlers import TimedRotatingFileHandler

from os.path import dirname, realpath

logger = logging.getLogger("")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s")
consoleAppender = logging.StreamHandler(sys.stdout)
consoleAppender.setFormatter(formatter)
logger.addHandler(consoleAppender)

fileAppender = TimedRotatingFileHandler(dirname(realpath(__file__)) + "/log/forex-expert-backend.log", 'D', 1, 5)
fileAppender.setFormatter(formatter)
logger.addHandler(fileAppender)

import os,sys
from pathlib import Path
import numpy as np

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

def logit(logstring,level=logging.INFO):
    if level==logging.INFO:
        logger.info(logstring)
    elif level==logging.DEBUG:
        logger.debug(logstring)
    elif level==logging.WARN:
        logger.warn(logstring)
    elif level==logging.ERROR:
        logger.error(logstring)
    else:
        logger.info(logstring)

def writeline(string, file):
    with open(file, 'a') as the_file:
        the_file.write('{}\n'.format(string))
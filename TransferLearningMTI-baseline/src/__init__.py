import logging
import os

logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
os.chdir('..')
os.environ['TV_CPP_MIN_LOG_LEVEL'] = '2'

import logging
import sys

logger = logging.getLogger("sb3_extra_buffers")
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))

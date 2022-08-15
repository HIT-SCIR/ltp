import logging


def get_pylogger(name=__name__) -> logging.Logger:
    logger = logging.getLogger(name)
    return logger

# Auto-generated stub
import logging
import os

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(name)s: %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

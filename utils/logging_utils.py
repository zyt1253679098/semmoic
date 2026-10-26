import logging
import os


def setup_logger(name='SemMoic', log_dir='logs', filename=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if filename is not None:
        os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(os.path.join(log_dir, filename))
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

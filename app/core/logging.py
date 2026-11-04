import logging

LOGGER_NAME = "moodi"


def configure_logging(level: str) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)

    for handler in logger.handlers:
        handler.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}")

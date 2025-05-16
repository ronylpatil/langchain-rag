# logger_utils.py
import logging

infologger = logging.getLogger("app_logger")
infologger.setLevel(logging.INFO)

if not infologger.hasHandlers():  # Avoid adding multiple handlers if imported multiple times
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(levelname)s:\t %(message)s"
    )
    console_handler.setFormatter(formatter)
    infologger.addHandler(console_handler)

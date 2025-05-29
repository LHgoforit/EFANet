# src/utils/logger.py

import os
import logging
import datetime


class ColorFormatter(logging.Formatter):
    """Custom formatter with color-coded log levels for terminal output."""
    COLORS = {
        'DEBUG': '\033[94m',   # Blue
        'INFO': '\033[92m',    # Green
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',   # Red
        'CRITICAL': '\033[1;91m',  # Bold Red
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        msg = super().format(record)
        return f"{color}{msg}{self.RESET}"


def setup_logger(name: str, save_dir: str = None, filename: str = 'log.txt', level=logging.INFO) -> logging.Logger:
    """
    Initialize and return a logger that writes to both stdout and an optional file.

    Args:
        name (str): Logger name.
        save_dir (str, optional): Directory to save the log file. If None, only console output.
        filename (str): Log file name.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent duplicate logs in notebooks

    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    color_formatter = ColorFormatter('[%(asctime)s] [%(levelname)s] %(message)s',
                                     datefmt='%Y-%m-%d %H:%M:%S')

    # Stream (console) handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, filename)
        file_handler = logging.FileHandler(file_path, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_timestamp() -> str:
    """
    Generate current timestamp in compact YYYYMMDD_HHMMSS format.

    Returns:
        str: Timestamp string.
    """
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

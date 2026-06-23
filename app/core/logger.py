"""
Project logging utilities.
Built with loguru, supports dual console/file output configured by `.env`,
and automatically generates `logs/app_YYYYMMDD.log`.
Features:
1. Configuration-driven: control output switches and log levels from `.env`
2. Automatic paths: file logs default to `project_root/logs/app_YYYYMMDD.log`
3. Automatic cleanup: retain logs by configuration and remove expired files
4. UTF-8 friendly: prevents garbled non-ASCII text
5. Async-safe: uses queued writes for multithreaded and async scenarios
6. Ready to use: every module can import `logger` directly
7. Accurate call sites: skips internal loguru frames and this utility layer
"""
import sys
import inspect
from pathlib import Path
import os
from dotenv import load_dotenv
from loguru import logger


# -------------------------- Step 1: Load .env --------------------------
load_dotenv()

# -------------------------- Step 2: Read .env settings --------------------------
LOG_CONSOLE_ENABLE = os.getenv("LOG_CONSOLE_ENABLE", "True").lower() == "true"
LOG_CONSOLE_LEVEL = os.getenv("LOG_CONSOLE_LEVEL", "INFO").upper()
LOG_FILE_ENABLE = os.getenv("LOG_FILE_ENABLE", "True").lower() == "true"
LOG_FILE_LEVEL = os.getenv("LOG_FILE_LEVEL", "INFO").upper()
LOG_FILE_RETENTION = os.getenv("LOG_FILE_RETENTION", "7 days")

# -------------------------- Step 3: Define log paths --------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE_NAME = "app_{time:YYYYMMDD}.log"
LOG_FILE_PATH = LOG_DIR / LOG_FILE_NAME

# -------------------------- Step 4: Define log format --------------------------
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name: <20}</cyan>:<cyan>{function: <15}</cyan>:<cyan>{line: <4}</cyan> - "
    "<level>{message}</level>"
)

# -------------------------- Step 5: Initialize logger --------------------------
def init_logger():
    """
    Initialize global logging configuration.
    1. Remove the default loguru console output
    2. Enable or disable console output based on `.env`
    3. Enable or disable file output based on `.env`
    4. Configure format, level, rotation, and retention
    :return: Configured loguru logger instance
    """
    # 1. Remove the default console sink from loguru.
    logger.remove()

    # 2. Configure console output when enabled.
    if LOG_CONSOLE_ENABLE:
        logger.add(
            sink=sys.stdout,
            level=LOG_CONSOLE_LEVEL,
            format=LOG_FORMAT,
            colorize=True,
            enqueue=True
        )

    # 3. Configure file output when enabled.
    if LOG_FILE_ENABLE:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.add(
            sink=LOG_FILE_PATH,
            level=LOG_FILE_LEVEL,
            format=LOG_FORMAT,
            rotation="00:00",
            retention=LOG_FILE_RETENTION,
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=True
        )

    return logger

# -------------------------- Step 6: Patch call-site reporting --------------------------
base_logger = init_logger()

def fix_log_position(record):
    """Walk the call stack and capture the actual business-code call site."""
    for frame in inspect.stack():
        # Skip internal loguru frames and this helper file.
        if ("_logger.py" in frame.filename or frame.function == "_log") or "logger.py" in frame.filename:
            continue
        # Update the record with the real calling location.
        record.update(
            name=frame.filename.split("/")[-1].split("\\")[-1],
            function=frame.function,
            line=frame.lineno
        )
        break

# Export the globally available logger with patched call-site resolution.
logger = base_logger.patch(fix_log_position)

# -------------------------- Local test --------------------------
if __name__ == '__main__':
    logger.info("[Test] Internal call from logger.py (business modules will show their own filename)")
    print(f"Log file output path: {LOG_FILE_PATH}")

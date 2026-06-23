# test/02-logger-test.py

from app.core.logger import logger

# --- 1. TRACE (most detailed) ---
# Use case: extremely detailed internal flow tracing, typically for debugging
# complex algorithms or state machines.
# Color: usually dark cyan / blue, depending on the terminal theme.
logger.trace("Entered function calculate_complex_logic with x=10, y=20")
logger.trace("Intermediate variable state={'step': 1, 'val': 30}")

# --- 2. DEBUG ---
# Use case: development-stage debug information, variable values, and function boundaries.
# Color: blue.
logger.debug("Current database connection pool size: 5")
logger.debug("Attempting retry request #2...")

# --- 3. INFO ---
# Use case: key points in normal business flow, user actions, service start/stop.
# Color: white / green.
logger.info("User ID 1001 logged in successfully")
logger.info("Order #9527 created, amount: CNY 299.00")
logger.info("System health check passed")

# --- 4. SUCCESS (Loguru-specific) ---
# Use case: explicitly mark that a long-running or important task finished well.
# Color: green.
logger.success("Data backup completed. File saved to /backup/2026-03-15.zip")
logger.success("Model training finished, accuracy reached 98.5%")

# --- 5. WARNING ---
# Use case: non-fatal issues, deprecated APIs, missing config values, pre-retry notices.
# Color: yellow / orange.
logger.warning("Config file is missing 'TIMEOUT'; using default value 30s")
logger.warning("API response time exceeded 2s; performance may be degrading")
logger.warning("User password strength is weak; consider changing it")

# --- 6. ERROR ---
# Use case: an operation failed, but the program can still continue.
# Color: red.
logger.error("Unable to connect to the Redis server: Connection refused")
logger.error("Failed to parse data for user ID 1002; skipping the record")

# --- 7. CRITICAL ---
# Use case: fatal errors where the program can no longer continue.
# Color: dark red / red background.
logger.critical("Disk space is full. No new data can be written and the system will stop")
logger.critical("Core encryption key is missing; security module initialization failed")

# --- Exception capture demo (automatically recorded at ERROR level) ---
@logger.catch
def divide(a, b):
    return a / b

# try:
#     divide(10, 0)
# except ZeroDivisionError:
#     # Manually record the exception stack trace.
#     logger.exception("Division by zero occurred; computation terminated")
    # Equivalent to: logger.opt(exception=True).error("Division by zero occurred; computation terminated")


divide(10, 0)

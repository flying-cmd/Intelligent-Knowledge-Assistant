# test/01-env-and-system-env-priority.py

import os
from dotenv import load_dotenv

# Load the .env file.
load_dotenv(override=True)

print(os.getenv("OPENAI_API_KEY"))

# Actual default behavior: `override=False`
# - If the system environment variable does not exist, use the value from `.env`
# - If the system environment variable already exists, the system value wins
# To let `.env` override system variables, pass `override=True` explicitly.
# load_dotenv(override=True)

# Example: assume the system has `MY_KEY=system_val` and `.env` has `MY_KEY=dotenv_val`.
print(os.getenv("MY_KEY"))
# load_dotenv() -> prints `system_val` (system variable has higher priority)
# load_dotenv(override=True) -> prints `dotenv_val` (`.env` overrides the system value)

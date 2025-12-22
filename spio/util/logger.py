"""Spio logger configuration."""

import os

from .env import env_var_is_true

env_value = os.environ.get("SPIO_LOGGER", "0")
if env_value.isdigit():
    log_level = int(env_value)
    if log_level < 0:
        raise ValueError("Log level cannot be negative.")
elif env_var_is_true("SPIO_LOGGER"):
    log_level = 1
else:
    log_level = 0

logger_enabled = log_level > 0
logger_verbose = log_level > 1

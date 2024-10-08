import os

TRUTHS = ["true", "1", "yes", "y", "t"]
env_value = os.environ.get("SPIO_LOGGER", "0").lower()

if env_value in TRUTHS:
    log_level = 1
elif env_value.isdigit():
    log_level = int(env_value)
else:
    log_level = 0

logger_enabled = log_level > 0
logger_verbose = log_level > 1

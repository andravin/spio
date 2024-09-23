import os

TRUTHS = ["true", "1", "yes", "y", "t"]
logger_enabled = os.environ.get("SPIO_LOGGER", "False").lower() in TRUTHS

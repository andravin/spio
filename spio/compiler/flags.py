"""Set default values of compiler settings from environment variables."""

import os
from contextvars import ContextVar
from ..util import env_var_is_true

DEFAULT_WORKERS = 1


default_lineinfo = env_var_is_true("SPIO_LINEINFO")
default_debug = env_var_is_true("SPIO_DEBUG")
workers = int(os.environ.get("SPIO_WORKERS", f"{DEFAULT_WORKERS}"))
default_count_instructions = env_var_is_true("SPIO_COUNT_INSTRUCTIONS")
default_disasm = env_var_is_true("SPIO_DISASM")

lineinfo = ContextVar("lineinfo", default=default_lineinfo)
debug = ContextVar("debug", default=default_debug)
count_instructions = ContextVar(
    "count_instructions", default=default_count_instructions
)
print_disasm = ContextVar("disasm", default=default_disasm)

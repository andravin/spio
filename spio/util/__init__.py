"""Utility functions used by the Spio library."""

from .load_parameter_set import _load_dataclasses_from_resource, load_parameter_set
from .math import divup, next_relative_prime
from .close import assert_all_close_with_acc_depth
from .interval_timer import IntervalTimer, Timer, time_function
from .device_info import get_formatted_device_name, get_formatted_arch, get_device_ordinal
from .cache_dir import get_cache_dir
from .class_names import get_full_name, get_full_name_with_underscores
from .parse_kwargs import ParseKwargs
from .parse_dataclass import parse_dataclass
from .logger import logger_enabled, logger_verbose, log_level
from .tensor_format import to_channels_last
from .memory_format import SixteenChannelsLast, check_channels_last

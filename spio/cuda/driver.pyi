"""Type stubs for the CUDA driver API."""

# pylint: disable=unused-argument

from typing import Tuple, List, Any, Optional

class FunctionAttributes:
    """Type stub for the FunctionAttributes class."""

    max_dynamic_shared_memory_size: Optional[int]
    preferred_shared_memory_carveout: Optional[int]

class DeviceAttributes:
    """Type stub for the DeviceAttributes class."""

    multiprocessor_count: int
    l2_cache_size: int
    name: Optional[str]
    compute_capability: Optional[Tuple[int, int]]
    max_shared_memory_per_block_optin: Optional[int]
    num_partitions_per_sm: Optional[int]

class Function:
    """Type stub for the Function class.

    The Function class represents a CUDA kernel function.
    """

    def get_attributes(self) -> FunctionAttributes:
        """Return the function attributes."""

    def set_attributes(self, attr: FunctionAttributes) -> None:
        """Set the function attributes."""

    def launch(
        self,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        args: List[Any],
        shared_mem_bytes: int = 0,
    ) -> None:
        """Launch the function with the given parameters."""

class Module:
    """Type stub for the Module class."""

    def load(self, fname):
        """Load a CUDA module from a file."""

    def unload(self) -> None:
        """Unload the module from the device."""

    def load_data(self, image):
        """Load a CUDA module from binary data."""

    def get_function(self, name):
        """Get a function from the CUDA module."""

def get_device_attributes(device_ordinal=0) -> DeviceAttributes:
    """Get the attributes of a CUDA device."""

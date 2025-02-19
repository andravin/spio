"""Implements the CheckerboardSpec class for tensor layout. Use with IndexSpec and TensorSpec."""

from .subindex_protocol import SubindexProtocol


class CheckerboardIndexSpec(SubindexProtocol):
    """CUDA / C++ code generator for checkerboard index classes."""

    def __init__(self, class_name: str = None, ranks: int = 8, **dims):
        self.class_name = class_name
        self.ranks = ranks
        self.dims = dims
        assert len(dims) == 2, "CheckerboardIndexSpec requires exactly two dimensions"
        items = list(dims.items())
        pairs_name, num_pairs = items[0]
        colors_name, num_colors = items[1]
        assert num_colors == 2, "CheckerboardIndexSpec requires exactly two colors"
        self.pairs_name = pairs_name
        self.num_pairs = num_pairs
        self.colors_name = colors_name

    @property
    def size(self) -> int:
        """Return the compound size of the fused dimensions."""
        return self.num_pairs * 2

    def generate(self) -> str:
        """Return the CUDA / C++ source code for the checkerboard index subclass."""
        return f"""
class {self.class_name} : public spio::CheckerboardIndex<{self.ranks}> {{
public:
    using Base = spio::CheckerboardIndex<{self.ranks}>;

    using Base::Base;

    static constexpr unsigned {self.pairs_name.upper()} = {self.num_pairs};
    static constexpr unsigned {self.colors_name.upper()} = 2;
    static constexpr unsigned size = {self.size};

    DEVICE constexpr int {self.pairs_name}() const {{ return Base::pair(); }}
    DEVICE constexpr int {self.colors_name}() const {{ return Base::color(); }}
}};
"""

    def generate_offset_function_declaration(
        self, return_type: str, function_name: str
    ) -> str:
        """Return the CUDA / C++ source code for the checkerboard offset function declaration."""
        return f"DEVICE constexpr {return_type} {function_name}(int {self.pairs_name}, int {self.colors_name}) const {{"

    def generate_offset_function_call(self) -> str:
        """Return the CUDA / C++ source code for the checkerboard offset function call."""
        return f"spio::CheckerboardIndex<{self.ranks}>::offset({self.pairs_name}, {self.colors_name})"


def _checkerboard_header():
    return """
#include "spio/checkerboard_index.h"
"""

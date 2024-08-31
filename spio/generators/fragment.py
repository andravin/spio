from dataclasses import dataclass
from typing import List


@dataclass
class FragmentSpec:
    class_name: str
    fragment_type: str
    row: str
    col: str

    def generate(self) -> str:
        return f"""
class {self.class_name} : public spio::{self.fragment_type} {{
    public:
        DEVICE static constexpr int {self.row}(int lane_id, int idx) {{ return row(lane_id, idx); }}
        DEVICE static constexpr int {self.col}2(int lane_id) {{ return col2(lane_id); }}
        DEVICE static constexpr int {self.col}(int lane_id) {{ return col(lane_id); }}
}};
"""


def generate_fragments(
    namespace: str, fragment_specs: List[FragmentSpec], header_file_name: str
) -> None:
    code = _fragment_header()
    code += _start_namespace(namespace)
    for fragment_spec in fragment_specs:
        code += fragment_spec.generate()
        code += "\n"
    code += _end_namespace()
    with open(header_file_name, "w") as file:
        file.write(code)


def _fragment_header() -> str:
    return f"""
#include "spio/mma.cuh"
"""


def _start_namespace(namespace: str) -> str:
    return f"""

namespace {namespace} {{
"""


def _end_namespace() -> str:
    return f"""
}}
"""

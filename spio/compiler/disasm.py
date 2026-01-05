"""Function for disassembling CUDA cubins and counting SASS instructions."""

import subprocess
import json
from tempfile import NamedTemporaryFile

from .cuda_paths import nvdisasm_full_path


def _disasm(cubin) -> dict:
    """Disassemble the given cubin and return the JSON output."""
    with NamedTemporaryFile(suffix=".cubin") as cubin_file:
        cubin_file.write(cubin)
        cubin_file.flush()
        nvdisasm = nvdisasm_full_path()
        args = [nvdisasm, "-json", cubin_file.name]
        r = subprocess.run(args, capture_output=True, check=True)

    output = r.stdout.decode("utf-8")

    # Parse JSON output and count SASS instructions in the specified kernel.
    data = json.loads(output)
    return data


def disasm(
    cubin,
    kernel_name: str,
    print_disasm: bool = False,
) -> int:
    """Return the number of SASS instructions in the given cubin.

    This is a high-level measure of efficient code generation. Regressions in
    the efficiency of address calculations often show up as increases in the
    instruction count.

    Does not count any trailing NOP instructions.
    Does count any branch instruction after the EXIT instruction.

    https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#json-format
    """
    data = _disasm(cubin)

    # The JSON structure is: [metadata, [function1, function2, ...]]
    functions = data[1]
    for func in functions:
        if func["function-name"] == kernel_name:
            instructions = func["sass-instructions"]
            # Strip trailing NOP instructions (alignment padding)
            while instructions and instructions[-1]["opcode"] == "NOP":
                instructions = instructions[:-1]
            num_instructions = len(instructions)
            if print_disasm:
                for i, instr in enumerate(instructions):
                    addr = instr.get("address", i * 16)
                    predicate = instr.get("predicate", "")
                    opcode = instr["opcode"]
                    operands = instr.get("operands", "")
                    print(
                        f"{i:4d}  0x{addr:04x}:  {predicate:>4s}  {opcode} {operands}"
                    )
            return num_instructions

    raise ValueError(f"Kernel '{kernel_name}' not found in cubin")

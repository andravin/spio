"""Functions for parsing torch.profile.profile results."""

import re


TIME_HEADER_FIELDS = [
    "CUDA_time_av",
    "CUDA_total",
    "CPU_time_avg",
    "Self_CUDA",
    "CPU_total",
    "Self_CPU",
]


def _preprocess_data_lines(lines):
    # Identify the header line and the first row of dashes to determine field widths
    num_dash_lines = 0
    field_widths = None
    header_fields = ""

    processed_lines = []
    for line in lines:
        if re.match(r"^-+", line):
            if num_dash_lines == 0:
                dash_line = line.strip()
                field_widths = [len(field) for field in dash_line.split(" ") if field]
            num_dash_lines += 1
            if num_dash_lines == 3:
                break
        else:
            if num_dash_lines == 1:
                # This is the header
                line = re.sub(r"(?<! ) (?! )", "_", line.rstrip())
            fields = []
            start = 0
            for idx, width in enumerate(field_widths):
                field = line[start : start + width].strip()
                if num_dash_lines == 1:
                    header_fields = fields
                else:
                    field_name = header_fields[idx]
                    if field_name in TIME_HEADER_FIELDS:
                        if field.endswith("us"):
                            time_val = (
                                float(field[:-2]) / 1000.0
                            )  # Convert microseconds to milliseconds
                        elif field.endswith("ms"):
                            time_val = float(field[:-2])  # Already in milliseconds
                        else:
                            time_val = float(
                                field
                            )  # Assume it's already in milliseconds
                            raise ValueError(f"Unexpected time unit inf field: {field}")
                        field = str(time_val)
                fields.append(field)
                start += width + 2  # +2 for the spaces between fields
            processed_line = ";".join(fields)
            processed_lines.append(processed_line)

    return processed_lines


def preprocess_data_string(data_str: str) -> str:
    """Preprocess a torch.profile.profile table from a string.

    Converts to a semicolon-separated format with time values in
    milliseconds.
    """
    lines = data_str.split("\n")
    return _preprocess_data_lines(lines)


def preprocess_data_file(file_path: str) -> str:
    """Preprocess a torch.profile.profile table from a file.

    Converts to a semicolon-separated format with time values in
    milliseconds.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        return _preprocess_data_lines(lines)

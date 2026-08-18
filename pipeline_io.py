"""
Small helpers for reading and writing pipeline tables.
"""

from pathlib import Path

import numpy as np

NUMERIC_FMT = "%.10g"
DESIGN_FMT = "%.6f"


def load_table(path):
    """Load a whitespace-delimited numeric table as a 2-D array."""
    path = Path(path)
    try:
        data = np.loadtxt(path, ndmin=2)
    except ValueError:
        data = np.loadtxt(path, ndmin=2, skiprows=1)
    except OSError as exc:
        raise FileNotFoundError(f"Could not read '{path}'") from exc
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return np.asarray(data, dtype=float)


def save_table(path, array, fmt=NUMERIC_FMT, header=None):
    """Write a numeric table; creates parent folders if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"fmt": fmt}
    if header is not None:
        kwargs["header"] = header
        kwargs["comments"] = ""
    np.savetxt(path, array, **kwargs)


def split_case_ids(values, n_vars):
    """Split optional leading case_id column from design-variable columns."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] == n_vars + 1:
        return values[:, 0].astype(int), values[:, 1:]
    if values.shape[1] == n_vars:
        return np.arange(values.shape[0], dtype=int), values
    raise ValueError(
        f"Expected {n_vars} variable columns or {n_vars + 1} with case_id, "
        f"got {values.shape[1]}"
    )


def append_design_row(path, values, case_id=None, fmt=DESIGN_FMT, write_case_id=True):
    """Append one design row to a text file."""
    values = np.asarray(values, dtype=float).ravel()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value_text = " ".join(fmt % value for value in values)
    with path.open("a", encoding="utf-8") as file:
        if write_case_id and case_id is not None:
            file.write(f"{int(case_id)} {value_text}\n")
        else:
            file.write(f"{value_text}\n")


def clear_file(path):
    """Truncate a file (create empty)."""
    Path(path).write_text("", encoding="utf-8")

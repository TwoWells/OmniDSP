# _generate_utils.py
# Shared utility functions and constants for C++ test reference data generation scripts.

import numpy as np
from pathlib import Path
import logging
import os  # Included for os.linesep

# --- Shared Configuration ---

# Base output directory for data files (e.g., ../data/ relative to this script)
# Assumes this file (_generate_utils.py) lives inside the 'scripts' directory
OUTPUT_BASE_DIR = Path(__file__).parent.parent / "data"

# Formatting for saved floating point numbers
PRECISION_FMT = "%.18e"

# Delimiter used in data files (space for real, and between real/imag pairs)
DELIMITER = " "


# Optional: Centralized logging configuration
# Can be called by individual scripts if they don't set up their own.
def setup_logging():
    """Configures basic logging."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# --- Shared Functions ---


def save_data(
    suite_name: str,
    test_case_name: str,
    purpose: str,
    type_suffix: str,
    data: np.ndarray,
):
    """
    Handles formatting and saving numpy data with a dimension header to the correct file.

    Args:
        suite_name: The name of the test suite (used as subdirectory name in data/).
        test_case_name: The base name corresponding to the GoogleTest test case.
        purpose: Describes the data's role (e.g., "input", "expected", "filter_coeffs").
        type_suffix: Indicates the data type (e.g., "d", "f", "cd", "cf").
        data: The numpy array containing the data to save.
    """
    filename = f"{test_case_name}_{purpose}_{type_suffix}.txt"
    suite_output_dir = OUTPUT_BASE_DIR / suite_name
    output_path = suite_output_dir / filename
    suite_output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    logging.info(f"Generating: {output_path.relative_to(OUTPUT_BASE_DIR.parent)}")

    # Determine dimensions and format header string
    header = ""
    if data.ndim == 1:
        # Handle empty vector case explicitly for header
        header = f"# {data.shape[0]}" if data.size > 0 else "# 0"
    elif data.ndim == 2:
        header = f"# {data.shape[0]}x{data.shape[1]}"
    else:
        # Currently only support 1D and 2D arrays
        raise ValueError(
            f"Unsupported array dimension {data.ndim} for {filename}. Only 1D or 2D supported."
        )

    # Format data array for saving based on type
    if np.iscomplexobj(data):
        # Reshape complex data to [real, imag] pairs for saving
        if data.ndim == 1:
            # Need to handle empty complex vector case
            formatted_data = (
                np.vstack((data.real, data.imag)).T
                if data.size > 0
                else np.empty((0, 2))
            )
        elif data.ndim == 2:  # Handle 2D complex matrix
            # Flatten data first, then stack real/imag
            data_flat = data.flatten()
            formatted_data = np.vstack((data_flat.real, data_flat.imag)).T
        else:
            # Should have been caught by dimension check
            raise ValueError("Unexpected complex data dimension during formatting.")
    elif np.issubdtype(data.dtype, np.number):
        # Real data - can be 1D or 2D
        formatted_data = data
    else:
        raise TypeError(f"Unsupported data type for saving: {data.dtype}")

    # Write header and data to file
    try:
        with open(
            output_path, "w", newline=os.linesep
        ) as f:  # Use os.linesep for consistency
            f.write(header + "\n")  # Write the header line first
            if formatted_data.size > 0:  # Only save data if array is not empty
                # np.savetxt handles 1D, 2D real, and Nx2 complex format correctly with space delimiter
                np.savetxt(f, formatted_data, fmt=PRECISION_FMT, delimiter=DELIMITER)
    except Exception as e:
        logging.error(f"Error saving file {output_path}: {e}")
        raise  # Re-raise exception after logging

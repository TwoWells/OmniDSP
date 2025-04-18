import numpy as np
import scipy.signal as sp_signal

# Removed os import
import logging

# Import shared utility functions and constants
from _generate_utils import save_data, setup_logging

# Configure logging (using the utility function)
setup_logging()

# --- Configuration ---
# SUITE_NAME is specific to this script
SUITE_NAME = "window"
# OUTPUT_DIR is determined within save_data using OUTPUT_BASE_DIR from utils
N = 1024  # Window size - MUST MATCH C++ TEST FIXTURE
# PRECISION_FMT and DELIMITER are now imported
KAISER_BETA = 8.0  # Beta for Kaiser window - MUST MATCH C++ TEST FIXTURE

# --- Helper Functions ---
# save_data function is now imported from _generate_utils.py

# --- Data Generation ---

logging.info(f"Generating Window reference data (N={N}) in data/{SUITE_NAME}/...")

window_types = {
    "Hann": "hann",
    "Hamming": "hamming",
    "Blackman": "blackman",
    "Kaiser": ("kaiser", KAISER_BETA),  # Tuple for windows needing parameters
    # Add other windows here (e.g., Bartlett, Flattop) if needed
}

for test_name_base, scipy_name_or_tuple in window_types.items():
    # Generate Double Precision
    window_d = sp_signal.get_window(scipy_name_or_tuple, N, fftbins=False).astype(
        np.float64
    )
    # Use imported save_data
    save_data(SUITE_NAME, f"{test_name_base}_D", "expected", "d", window_d)

    # Generate Float Precision
    window_f = window_d.astype(np.float32)
    # Use imported save_data
    save_data(SUITE_NAME, f"{test_name_base}_F", "expected", "f", window_f)


logging.info("Window reference data generation complete.")

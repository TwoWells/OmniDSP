import numpy as np
import scipy.fft as sp_fft  # Use newer scipy.fft

# Removed os import
import logging

# Import shared utility functions and constants
from _generate_utils import save_data, setup_logging

# Configure logging (using the utility function)
setup_logging()

# --- Configuration ---
# SUITE_NAME is specific to this script
SUITE_NAME = "fft"
# OUTPUT_DIR is determined within save_data using OUTPUT_BASE_DIR from utils
N = 1024  # Signal length for most tests
# PRECISION_FMT and DELIMITER are now imported

# --- Helper Functions ---
# save_data function is now imported from _generate_utils.py

# --- Data Generation ---

logging.info(f"Generating FFT reference data (N={N}) in data/{SUITE_NAME}/...")

# Create base input signals
np.random.seed(42)  # for reproducibility
fft_input_real_d = np.random.randn(N)
fft_input_real_f = fft_input_real_d.astype(np.float32)
# Create complex input from real (simple example)
fft_input_complex_d = fft_input_real_d + 1j * np.random.randn(N)
fft_input_complex_f = fft_input_complex_d.astype(np.complex64)

# --- FFT (Complex -> Complex) ---
test_cases_fft = [
    ("Plan_FFT_Forward", "forward"),
    ("Plan_FFT_Backward", "backward"),
    ("Plan_FFT_Ortho", "ortho"),
    ("Convenience_FFT", "forward"),  # Convenience uses default forward
]
for base_name, norm in test_cases_fft:
    # Double
    expected_cd = sp_fft.fft(fft_input_complex_d, norm=norm)
    # Use imported save_data
    save_data(SUITE_NAME, f"{base_name}_Double", "input", "cd", fft_input_complex_d)
    save_data(SUITE_NAME, f"{base_name}_Double", "expected", "cd", expected_cd)
    # Float
    expected_cf = sp_fft.fft(fft_input_complex_f, norm=norm)
    # Use imported save_data
    save_data(SUITE_NAME, f"{base_name}_Float", "input", "cf", fft_input_complex_f)
    save_data(SUITE_NAME, f"{base_name}_Float", "expected", "cf", expected_cf)


# --- IFFT (Complex -> Complex) ---
test_cases_ifft = [
    ("Plan_IFFT_Forward", "forward"),
    ("Plan_IFFT_Backward", "backward"),
    ("Plan_IFFT_Ortho", "ortho"),
    ("Convenience_IFFT", "backward"),  # Convenience uses default backward
]
# Use FFT outputs as IFFT inputs for round-trip checks where appropriate
fft_forward_output_d = sp_fft.fft(fft_input_complex_d, norm="forward")
fft_forward_output_f = sp_fft.fft(fft_input_complex_f, norm="forward")

for base_name, norm in test_cases_ifft:
    # Double (Using forward FFT output as input for IFFT)
    input_cd = fft_forward_output_d  # Example input
    expected_cd = sp_fft.ifft(input_cd, norm=norm)
    # Use imported save_data
    save_data(SUITE_NAME, f"{base_name}_Double", "input", "cd", input_cd)
    save_data(SUITE_NAME, f"{base_name}_Double", "expected", "cd", expected_cd)
    # Float
    input_cf = fft_forward_output_f  # Example input
    expected_cf = sp_fft.ifft(input_cf, norm=norm)
    # Use imported save_data
    save_data(SUITE_NAME, f"{base_name}_Float", "input", "cf", input_cf)
    save_data(SUITE_NAME, f"{base_name}_Float", "expected", "cf", expected_cf)


# --- RFFT (Real -> Complex) ---
test_cases_rfft = [
    ("Plan_RFFT_Forward", "forward"),
    # Add Backward/Ortho if needed by tests
    ("Convenience_RFFT", "forward"),  # Convenience uses default forward
]
for base_name, norm in test_cases_rfft:
    # Double
    expected_cd = sp_fft.rfft(fft_input_real_d, norm=norm)
    # Use imported save_data
    save_data(SUITE_NAME, f"{base_name}_Double", "input", "d", fft_input_real_d)
    save_data(SUITE_NAME, f"{base_name}_Double", "expected", "cd", expected_cd)
    # Float
    expected_cf = sp_fft.rfft(fft_input_real_f, norm=norm)
    # Use imported save_data
    save_data(SUITE_NAME, f"{base_name}_Float", "input", "f", fft_input_real_f)
    save_data(SUITE_NAME, f"{base_name}_Float", "expected", "cf", expected_cf)


# --- IRFFT (Complex -> Real) ---
test_cases_irfft = [
    ("Plan_IRFFT_Forward", "forward"),
    ("Plan_IRFFT_Backward", "backward"),
    ("Plan_IRFFT_Ortho", "ortho"),
    ("Convenience_IRFFT", "backward"),  # Convenience uses default backward
]
# Use RFFT outputs as IRFFT inputs
rfft_forward_output_d = sp_fft.rfft(fft_input_real_d, norm="forward")
rfft_forward_output_f = sp_fft.rfft(fft_input_real_f, norm="forward")

for base_name, norm in test_cases_irfft:
    # Double
    input_cd = rfft_forward_output_d
    expected_d = sp_fft.irfft(input_cd, n=N, norm=norm)  # Specify N for irfft
    # Use imported save_data
    save_data(SUITE_NAME, f"{base_name}_Double", "input", "cd", input_cd)
    save_data(SUITE_NAME, f"{base_name}_Double", "expected", "d", expected_d)
    # Float
    input_cf = rfft_forward_output_f
    expected_f = sp_fft.irfft(input_cf, n=N, norm=norm)  # Specify N for irfft
    # Use imported save_data
    save_data(SUITE_NAME, f"{base_name}_Float", "input", "cf", input_cf)
    save_data(SUITE_NAME, f"{base_name}_Float", "expected", "f", expected_f)


logging.info("FFT reference data generation complete.")

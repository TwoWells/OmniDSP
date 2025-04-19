import numpy as np

# Use librosa for reference CQT calculation if available, otherwise use scipy/numpy basics
try:
    import librosa

    HAVE_LIBROSA = True
except ImportError:
    HAVE_LIBROSA = False
    # Define dummy functions or fallback logic if needed without librosa

import scipy.signal as sp_signal
import logging

# Import shared utility functions and constants
# Ensure _generate_utils.py is in the same directory or Python path
from _generate_utils import save_data, setup_logging

# Configure logging (using the utility function)
setup_logging()
if not HAVE_LIBROSA:
    logging.warning(
        "Librosa not found. CQT reference generation will be limited or use basic/dummy methods."
    )


# --- Configuration ---
# SUITE_NAME is specific to this script
SUITE_NAME = "cqt"
# OUTPUT_DIR is determined within save_data using OUTPUT_BASE_DIR from utils

# CQT Parameters (Match C++ tests)
SR = 22050.0
FMIN = 32.7  # C1
N_BINS = 84  # Example, might need adjustment based on test parameters
BINS_PER_OCTAVE = 12
# Derived N_OCTAVES = N_BINS // BINS_PER_OCTAVE (adjust N_BINS if N_OCTAVES is fixed)
# Example: If C++ test uses 5 octaves starting from C1
# FMAX calculation based on N_BINS and FMIN isn't directly needed if N_BINS is primary
# HIGH_FREQ = FMIN * (2.0 ** (N_BINS / BINS_PER_OCTAVE)) # Calculate if needed

# Signal Parameters
N_SIGNAL = 4096  # Example signal length
FREQ_SIGNAL = 440.0  # A4

# Resampling Parameters (for FilterAndDownsample test)
N_RESAMPLE_INPUT = 2048
FILTER_ORDER = 100  # Example filter order - MUST match C++ test FIR generation if used

# --- Helper Functions ---
# save_data function is now imported from _generate_utils.py

# --- Data Generation ---

logging.info(f"Generating CQT reference data in data/{SUITE_NAME}/...")

# 1. Basic Input Signal (Real for Librosa CQT, Complex potentially for OmniDSP test)
np.random.seed(42)  # for reproducibility
t = np.arange(N_SIGNAL) / SR
signal_real_d = 0.5 * np.sin(2 * np.pi * FREQ_SIGNAL * t)
signal_real_f = signal_real_d.astype(np.float32)

# Save real versions which are likely inputs for CQT execute tests
save_data(SUITE_NAME, "FullRecursiveCQT_Execute_Double", "input", "d", signal_real_d)
save_data(SUITE_NAME, "FullRecursiveCQT_Execute_Float", "input", "f", signal_real_f)


# 2. Expected CQT Output (using Librosa if available)
# This section generates reference based on parameters defined above
cqt_test_case_name = "FullRecursiveCQT_Execute"  # Base name for CQT test output

if HAVE_LIBROSA:
    logging.info("Generating Librosa CQT reference...")
    try:
        # Librosa CQT returns complex matrix [n_bins, n_frames]
        # Use hop_length consistent with potential C++ test needs if applicable
        # Default hop length in librosa.cqt is often derived from filter lengths.
        # If C++ test uses fixed hop, specify it here if possible.
        hop_length_cqt = 512  # Example hop_length for librosa call
        cqt_librosa_d = librosa.cqt(
            y=signal_real_d,  # Use real input
            sr=SR,
            fmin=FMIN,
            n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE,
            hop_length=hop_length_cqt,  # Specify hop length if needed
            # Add other relevant parameters like window, filter_scale etc. if needed for match
        )
        # C++ test expects a 2D matrix [Bins x Frames]
        expected_cqt_d = cqt_librosa_d  # Keep as 2D matrix
        expected_cqt_f = expected_cqt_d.astype(np.complex64)

        save_data(
            SUITE_NAME, f"{cqt_test_case_name}_Double", "expected", "cd", expected_cqt_d
        )
        save_data(
            SUITE_NAME, f"{cqt_test_case_name}_Float", "expected", "cf", expected_cqt_f
        )

    except Exception as e:
        logging.error(f"Error generating Librosa CQT: {e}. Saving zeros instead.")
        # Determine expected frames based on signal length and hop length
        num_frames_approx = (N_SIGNAL + hop_length_cqt - 1) // hop_length_cqt
        expected_cqt_d = np.zeros((N_BINS, num_frames_approx), dtype=np.complex128)
        expected_cqt_f = np.zeros((N_BINS, num_frames_approx), dtype=np.complex64)
        save_data(
            SUITE_NAME, f"{cqt_test_case_name}_Double", "expected", "cd", expected_cqt_d
        )
        save_data(
            SUITE_NAME, f"{cqt_test_case_name}_Float", "expected", "cf", expected_cqt_f
        )
else:
    logging.warning("Librosa not found, saving zeros for CQT expected values.")
    # Determine approximate frame count needed for zero matrix
    hop_length_cqt = 512  # Assume default hop length if librosa not available
    num_frames_approx = (N_SIGNAL + hop_length_cqt - 1) // hop_length_cqt
    expected_cqt_d = np.zeros((N_BINS, num_frames_approx), dtype=np.complex128)
    expected_cqt_f = np.zeros((N_BINS, num_frames_approx), dtype=np.complex64)
    save_data(
        SUITE_NAME, f"{cqt_test_case_name}_Double", "expected", "cd", expected_cqt_d
    )
    save_data(
        SUITE_NAME, f"{cqt_test_case_name}_Float", "expected", "cf", expected_cqt_f
    )


# 3. Filter Bank (for Precomputed Test) - Dummy Data Generation
# This section remains as placeholder/example
# logging.warning("Generating DUMMY CQT filter bank data.")
# N_FFT_EXAMPLE = 2048 # Adjust if needed
# dummy_filter_bank_d = np.random.randn(N_BINS, N_FFT_EXAMPLE // 2 + 1).astype(np.complex128)
# save_data(SUITE_NAME, "Precomputed_FilterBank", "expected", "cd", dummy_filter_bank_d)


# 4. Data for FilterAndDownsample Test
logging.info("Generating Resample (FilterAndDownsample) DOUBLE reference data...")
resample_input_d = np.random.randn(N_RESAMPLE_INPUT)
# Create simple FIR filter coefficients (e.g., lowpass) matching C++ test expectations
# Use a cutoff slightly less than Nyquist/2 (which is 0.5*fs / 2 = 0.25*fs -> normalized 0.25)
# Ensure the length (FILTER_ORDER + 1) matches C++ test usage
resample_filter_coeffs_d = sp_signal.firwin(
    FILTER_ORDER + 1,
    0.45,
    window="hamming",  # Cutoff relative to Nyquist (fs/2)
).astype(np.float64)
# Perform reference calculation (convolve then downsample)
filtered_d = sp_signal.convolve(
    resample_input_d, resample_filter_coeffs_d, mode="valid"
)
resample_expected_d = filtered_d[::2]  # Downsample by 2

# Save double precision versions
resample_test_case_name = "FilterAndDownsample"
save_data(SUITE_NAME, resample_test_case_name, "input", "d", resample_input_d)
save_data(
    SUITE_NAME, resample_test_case_name, "filter_coeffs", "d", resample_filter_coeffs_d
)
save_data(SUITE_NAME, resample_test_case_name, "expected", "d", resample_expected_d)

# === Add this block to generate FLOAT reference data ===
logging.info("Generating Resample (FilterAndDownsample) FLOAT reference data...")
# Convert inputs to float32
resample_input_f = resample_input_d.astype(np.float32)
resample_filter_coeffs_f = resample_filter_coeffs_d.astype(np.float32)
# Perform reference calculation using float32
filtered_f = sp_signal.convolve(
    resample_input_f, resample_filter_coeffs_f, mode="valid"
)
resample_expected_f = filtered_f[::2]  # Downsample by 2
# Use imported save_data to save the float versions
save_data(SUITE_NAME, resample_test_case_name, "input", "f", resample_input_f)
save_data(
    SUITE_NAME, resample_test_case_name, "filter_coeffs", "f", resample_filter_coeffs_f
)
save_data(SUITE_NAME, resample_test_case_name, "expected", "f", resample_expected_f)
logging.info("FLOAT Resample reference data generation complete.")
# === End of block to add ===


logging.info("CQT and related reference data generation complete.")

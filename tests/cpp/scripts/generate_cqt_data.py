import numpy as np

# Use librosa for reference CQT calculation if available, otherwise use scipy/numpy basics
try:
    import librosa

    HAVE_LIBROSA = True
except ImportError:
    HAVE_LIBROSA = False
    # Define dummy functions or fallback logic if needed without librosa

import scipy.signal as sp_signal

# Removed os import as it's not directly needed here anymore
import logging

# Import shared utility functions and constants
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
N_BINS = 84
BINS_PER_OCTAVE = 12
N_OCTAVES = N_BINS // BINS_PER_OCTAVE

# Signal Parameters
N_SIGNAL = 4096  # Example signal length
FREQ_SIGNAL = 440.0  # A4

# Resampling Parameters (for FilterAndDownsample test)
N_RESAMPLE_INPUT = 2048
FILTER_ORDER = 100  # Example filter order

# --- Helper Functions ---
# save_data function is now imported from _generate_utils.py

# --- Data Generation ---

logging.info(f"Generating CQT reference data in data/{SUITE_NAME}/...")

# 1. Basic Input Signal (Complex)
np.random.seed(42)  # for reproducibility
t = np.arange(N_SIGNAL) / SR
signal_real_d = 0.5 * np.sin(2 * np.pi * FREQ_SIGNAL * t)
signal_real_f = signal_real_d.astype(np.float32)
# Create complex input (e.g., analytic signal or just add noise)
# Placeholder: CQT often uses real input, but tests might need complex
signal_complex_d = signal_real_d + 0j
signal_complex_f = signal_complex_d.astype(np.complex64)

# Use imported save_data
save_data(
    SUITE_NAME, "FullRecursiveCQT_Execute_Double", "input", "cd", signal_complex_d
)
save_data(SUITE_NAME, "FullRecursiveCQT_Execute_Float", "input", "cf", signal_complex_f)
# Also save for precomputed test if needed
# save_data(SUITE_NAME, "Precomputed_Execute", "input", "cd", signal_complex_d)

# 2. Expected CQT Output (using Librosa if available)
if HAVE_LIBROSA:
    logging.info("Generating Librosa CQT reference...")
    # Librosa CQT returns complex matrix [n_bins, n_frames]
    # NOTE: Librosa CQT might require different padding/framing than OmniDSP's direct execution.
    # This reference might only be approximate or need careful alignment.
    # The C++ test currently expects a 1D vector, so we save only the first frame.
    # Adjust this if the C++ implementation changes output format.
    try:
        # Ensure consistent precision for librosa input if needed
        cqt_librosa_d = librosa.cqt(
            y=signal_real_d.astype(np.float64),
            sr=SR,
            fmin=FMIN,
            n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE,
        )
        cqt_librosa_f = cqt_librosa_d.astype(np.complex64)
        # Taking first frame as example for 1D output test
        expected_cqt_d = cqt_librosa_d[:, 0]
        # Taking first frame as example for 1D output test
        expected_cqt_f = cqt_librosa_f[:, 0]
        # Use imported save_data
        save_data(
            SUITE_NAME,
            "FullRecursiveCQT_Execute_Double",
            "expected",
            "cd",
            expected_cqt_d,
        )
        save_data(
            SUITE_NAME,
            "FullRecursiveCQT_Execute_Float",
            "expected",
            "cf",
            expected_cqt_f,
        )
        # save_data(SUITE_NAME, "Precomputed_Execute", "expected", "cd", expected_cqt_d) # If needed
    except Exception as e:
        logging.error(f"Error generating Librosa CQT: {e}. Saving zeros instead.")
        expected_cqt_d = np.zeros(N_BINS, dtype=np.complex128)
        expected_cqt_f = np.zeros(N_BINS, dtype=np.complex64)
        # Use imported save_data
        save_data(
            SUITE_NAME,
            "FullRecursiveCQT_Execute_Double",
            "expected",
            "cd",
            expected_cqt_d,
        )
        save_data(
            SUITE_NAME,
            "FullRecursiveCQT_Execute_Float",
            "expected",
            "cf",
            expected_cqt_f,
        )

else:
    logging.warning("Librosa not found, saving zeros for CQT expected values.")
    expected_cqt_d = np.zeros(N_BINS, dtype=np.complex128)
    expected_cqt_f = np.zeros(N_BINS, dtype=np.complex64)
    # Use imported save_data
    save_data(
        SUITE_NAME, "FullRecursiveCQT_Execute_Double", "expected", "cd", expected_cqt_d
    )
    save_data(
        SUITE_NAME, "FullRecursiveCQT_Execute_Float", "expected", "cf", expected_cqt_f
    )


# 3. Filter Bank (for Precomputed Test)
# Generate dummy filter bank data (2D complex)
# TODO: Replace with actual filter bank generation if needed.
logging.warning("Generating DUMMY CQT filter bank data.")
# Example shape - adjust N_FFT based on actual CQT implementation needs
N_FFT_EXAMPLE = 2048
dummy_filter_bank_d = np.random.randn(N_BINS, N_FFT_EXAMPLE // 2 + 1).astype(
    np.complex128
)
# Use imported save_data
save_data(SUITE_NAME, "Precomputed_FilterBank", "expected", "cd", dummy_filter_bank_d)


# 4. Data for FilterAndDownsample Test
logging.info("Generating Resample (FilterAndDownsample) reference data...")
resample_input_d = np.random.randn(N_RESAMPLE_INPUT)
# Create simple FIR filter coefficients (e.g., lowpass)
# Note: These coeffs MUST match what the C++ code expects/uses for this test
resample_filter_coeffs_d = sp_signal.firwin(
    FILTER_ORDER + 1, 0.45, window="hamming"
)  # Cutoff relative to Nyquist
# Perform reference calculation (convolve then downsample)
filtered = sp_signal.convolve(resample_input_d, resample_filter_coeffs_d, mode="valid")
# Downsample by 2 (simple slicing)
resample_expected_d = filtered[::2]

# Use imported save_data
save_data(SUITE_NAME, "FilterAndDownsample", "input", "d", resample_input_d)
save_data(
    SUITE_NAME, "FilterAndDownsample", "filter_coeffs", "d", resample_filter_coeffs_d
)
save_data(SUITE_NAME, "FilterAndDownsample", "expected", "d", resample_expected_d)


logging.info("CQT and related reference data generation complete.")

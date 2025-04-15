# tests/cpp/generate_references.py
# Expanded to include Librosa CQT and SciPy Filter+Downsample reference data

import numpy as np
from scipy import signal
# --- Added firwin and correlate ---
from scipy.signal import firwin, correlate
import librosa
import os
import warnings

# --- Define Input Signals and Kernels (for Conv/Corr tests) ---
signal_d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float64)
kernel_d = np.array([0.5, 1.0, 0.5], dtype=np.float64)        # Symmetric
kernel_d_asym = np.array([1.0, 2.0, 0.5], dtype=np.float64)   # Asymmetric

signal_f = signal_d.astype(np.float32)
kernel_f = kernel_d.astype(np.float32)
kernel_f_asym = kernel_d_asym.astype(np.float32)

signal_edge = np.array([1.0, 2.0, 3.0], dtype=np.float64)
kernel_edge = np.array([0.5, 1.0, 0.5], dtype=np.float64)

# --- Calculate Expected Results (for Conv/Corr tests) ---
expected_corr_d = signal.correlate(signal_d, kernel_d, mode='valid')
expected_corr_f = signal.correlate(signal_f, kernel_f, mode='valid')
expected_corr_edge_d = signal.correlate(signal_edge, kernel_edge, mode='valid')

expected_conv_d = signal.convolve(signal_d, kernel_d_asym, mode='valid')
expected_conv_f = signal.convolve(signal_f, kernel_f_asym, mode='valid')
expected_conv_edge_d = signal.convolve(signal_edge, kernel_edge, mode='valid')

# --- REMOVED old desamp calculations - will recalculate below ---
# factor = 2
# filtered_d = signal.correlate(signal_d, kernel_d, mode='valid')
# expected_desamp_d = filtered_d[::factor]
# filtered_f = signal.correlate(signal_f, kernel_f, mode='valid')
# expected_desamp_f = filtered_f[::factor]
# filtered_edge_d = signal.correlate(signal_edge, kernel_edge, mode='valid')
# expected_desamp_edge_d = filtered_edge_d[::factor]
#
# factor3 = 3
# filtered_d_f3 = signal.correlate(signal_d, kernel_d, mode='valid')
# expected_desamp_d_f3 = filtered_d_f3[::factor3]

# --- Define Parameters and Signal for CQT Reference ---
# Match parameters used in C++ tests (PrecomputedRecursiveCQTTest fixture)
cqt_sr = 44100.0
cqt_hop_length = 512
cqt_fmin = 55.0   # A1
cqt_fmax = 1760.0 # A6
cqt_bins_per_octave = 12
cqt_n_bins = int(np.ceil(cqt_bins_per_octave * np.log2(cqt_fmax / cqt_fmin)))
cqt_signal_freq = 440.0
cqt_signal_duration = 1.0
cqt_signal_len = int(cqt_signal_duration * cqt_sr)
cqt_time = np.arange(cqt_signal_len) / cqt_sr
cqt_signal_d = np.sin(2 * np.pi * cqt_signal_freq * cqt_time).astype(np.float64)

# --- Calculate Librosa CQT Reference ---
print(f"\nCalculating Librosa CQT reference...")
print(f"  Parameters: sr={cqt_sr}, hop_length={cqt_hop_length}, fmin={cqt_fmin}, n_bins={cqt_n_bins}, bins_per_octave={cqt_bins_per_octave}")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    librosa_cqt_complex_d = librosa.cqt(
        y=cqt_signal_d, sr=cqt_sr, hop_length=cqt_hop_length, fmin=cqt_fmin,
        n_bins=cqt_n_bins, bins_per_octave=cqt_bins_per_octave,
        filter_scale=1, res_type='kaiser_fast', dtype=np.complex128
    )
print(f"  Librosa CQT calculated. Shape: {librosa_cqt_complex_d.shape}")

# --- Calculate Filter + Downsample Reference Data ---
print("\nCalculating Filter+Downsample reference data...")
filter_downsample_sr = 4000.0
filter_downsample_len = 2048
filter_downsample_order = 101 # Matches DEFAULT_FIR_FILTER_ORDER_TEST
filter_downsample_cutoff = filter_downsample_sr / 4.0 # Matches C++ test
downsample_factor = 2
filter_downsample_freq_low = 500.0
filter_downsample_freq_high = 750.0 # Matches C++ test

# Generate test signal (matches C++ test)
filter_downsample_time = np.arange(filter_downsample_len) / filter_downsample_sr
signal_filter_test_d = (0.5 * np.sin(2 * np.pi * filter_downsample_freq_low * filter_downsample_time) +
                        0.5 * np.sin(2 * np.pi * filter_downsample_freq_high * filter_downsample_time)).astype(np.float64)
signal_filter_test_f = signal_filter_test_d.astype(np.float32)

# Design FIR filter coefficients (matches C++ test's windowed-sinc design)
# Use firwin with scale=True to approximate the DC gain normalization (sum=1)
coeffs_d = firwin(filter_downsample_order,
                  cutoff=filter_downsample_cutoff,
                  window='hann',
                  fs=filter_downsample_sr,
                  scale=True) # scale=True sets gain at DC (0 Hz) to 1.0
coeffs_f = coeffs_d.astype(np.float32)

print(f"  FIR filter designed (Order: {filter_downsample_order}, Cutoff: {filter_downsample_cutoff} Hz, Window: Hann)")
# print(f"  Filter coefficients sum (double): {np.sum(coeffs_d)}") # Should be close to 1.0

# Perform correlation (matches C++ backend FIR filtering) in 'valid' mode
filtered_d = correlate(signal_filter_test_d, coeffs_d, mode='valid')
filtered_f = correlate(signal_filter_test_f, coeffs_f, mode='valid')

# Perform downsampling
expected_filter_downsample_d = filtered_d[::downsample_factor]
expected_filter_downsample_f = filtered_f[::downsample_factor]

print(f"  Filter+Downsample calculated. Output shape (double): {expected_filter_downsample_d.shape}")
print(f"  Filter+Downsample calculated. Output shape (float): {expected_filter_downsample_f.shape}")


# --- Data to Write ---
data_to_write = {
    # Conv/Corr data
    "expected_corr_d": expected_corr_d,
    "expected_corr_f": expected_corr_f,
    "expected_corr_edge_d": expected_corr_edge_d,
    "expected_conv_d": expected_conv_d,
    "expected_conv_f": expected_conv_f,
    "expected_conv_edge_d": expected_conv_edge_d,
    # Filter+Downsample data (NEW)
    "expected_filter_downsample_d": expected_filter_downsample_d,
    "expected_filter_downsample_f": expected_filter_downsample_f,
    # Librosa CQT data
    "librosa_cqt_complex_d": librosa_cqt_complex_d,
}

# --- Write to File ---
output_filename = "test_references.txt"
print(f"\nWriting reference data to {output_filename}...")

try:
    with open(output_filename, "w") as f:
        f.write("# Auto-generated reference data for OmniDSP C++ tests\n")
        f.write("# Includes Conv/Corr (from SciPy), Filter+Downsample (from SciPy), and CQT (from Librosa)\n\n") # Updated comment

        for name, arr in data_to_write.items():
            # Determine type for formatting
            is_float = arr.dtype == np.float32
            is_complex = np.iscomplexobj(arr)
            is_2d = arr.ndim == 2

            # Write start marker
            f.write(f"# START {name}\n")

            # Add shape information for 2D arrays
            if is_2d:
                f.write(f"# SHAPE: {arr.shape[0]} {arr.shape[1]}\n")
                arr_flat = arr.flatten()
            else:
                arr_flat = arr

            # Write data
            for val in arr_flat:
                if is_complex:
                    if isinstance(val.real, np.float32):
                        f.write(f"{val.real:.9g}f # real\n")
                        f.write(f"{val.imag:.9g}f # imag\n")
                    else:
                        f.write(f"{val.real:.17g} # real\n")
                        f.write(f"{val.imag:.17g} # imag\n")
                elif is_float:
                    f.write(f"{val:.9g}f\n")
                else: # Double real
                    f.write(f"{val:.17g}\n")

            # Write end marker
            f.write(f"# END {name}\n\n")

    print("Finished writing reference data.")

except ImportError as e:
    print(f"\nError: Required library not found ({e}). Please ensure SciPy and Librosa are installed.")
    print("You can typically install them using: conda install scipy librosa -c conda-forge")
except IOError as e:
    print(f"Error writing file {output_filename}: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


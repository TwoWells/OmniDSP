# tests/cpp/generate_references.py
# Expanded to include Librosa CQT reference data

import numpy as np
from scipy import signal
import librosa  # <-- Added Librosa import
import os
import warnings  # To potentially ignore librosa warnings if needed

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

factor = 2
filtered_d = signal.correlate(signal_d, kernel_d, mode='valid')
expected_desamp_d = filtered_d[::factor]
filtered_f = signal.correlate(signal_f, kernel_f, mode='valid')
expected_desamp_f = filtered_f[::factor]
filtered_edge_d = signal.correlate(signal_edge, kernel_edge, mode='valid')
expected_desamp_edge_d = filtered_edge_d[::factor]

factor3 = 3
filtered_d_f3 = signal.correlate(signal_d, kernel_d, mode='valid')
expected_desamp_d_f3 = filtered_d_f3[::factor3]

# --- Define Parameters and Signal for CQT Reference ---
# Match parameters used in C++ tests (PrecomputedRecursiveCQTTest fixture)
cqt_sr = 44100.0
cqt_hop_length = 512
cqt_fmin = 55.0   # A1
# A6 (Note: Librosa calculates bins up to fmax, OmniDSP might include bins slightly above)
cqt_fmax = 1760.0
cqt_bins_per_octave = 12
# Calculate n_bins based on the range from fmin to fmax
cqt_n_bins = int(np.ceil(cqt_bins_per_octave * np.log2(cqt_fmax / cqt_fmin)))

# Create a test signal (e.g., 1 second sine wave at 440 Hz)
cqt_signal_freq = 440.0  # A4
cqt_signal_duration = 1.0  # seconds
cqt_signal_len = int(cqt_signal_duration * cqt_sr)
cqt_time = np.arange(cqt_signal_len) / cqt_sr
# Use float64 for reference calculation
cqt_signal_d = np.sin(2 * np.pi * cqt_signal_freq *
                      cqt_time).astype(np.float64)

# --- Calculate Librosa CQT Reference ---
print(f"\nCalculating Librosa CQT reference...")
print(f"  Parameters: sr={cqt_sr}, hop_length={cqt_hop_length}, fmin={cqt_fmin}, n_bins={cqt_n_bins}, bins_per_octave={cqt_bins_per_octave}")

# Use filter_scale=1 for potentially closer match to basic windowed FFT approach
# res_type='kaiser_fast' is default and efficient
# Ensure dtype is complex128 for double precision reference
with warnings.catch_warnings():  # Suppress potential Numba/audioread warnings
    warnings.simplefilter("ignore")
    librosa_cqt_complex_d = librosa.cqt(
        y=cqt_signal_d,
        sr=cqt_sr,
        hop_length=cqt_hop_length,
        fmin=cqt_fmin,
        n_bins=cqt_n_bins,  # Total number of bins
        bins_per_octave=cqt_bins_per_octave,
        filter_scale=1,  # Adjust if needed based on comparison
        res_type='kaiser_fast',
        dtype=np.complex128  # Ensure double precision complex output
    )
# Shape: (n_bins, n_frames)
print(f"  Librosa CQT calculated. Shape: {librosa_cqt_complex_d.shape}")

# --- Data to Write ---
# Store data in a dictionary for easier iteration
data_to_write = {
    # Existing Conv/Corr/Desamp data
    "expected_corr_d": expected_corr_d,
    "expected_corr_f": expected_corr_f,
    "expected_corr_edge_d": expected_corr_edge_d,
    "expected_conv_d": expected_conv_d,
    "expected_conv_f": expected_conv_f,
    "expected_conv_edge_d": expected_conv_edge_d,
    "expected_desamp_d": expected_desamp_d,
    "expected_desamp_f": expected_desamp_f,
    "expected_desamp_edge_d": expected_desamp_edge_d,
    "expected_desamp_d_f3": expected_desamp_d_f3,
    # New Librosa CQT data
    "librosa_cqt_complex_d": librosa_cqt_complex_d,  # Store the complex result
}

# --- Write to File ---
output_filename = "test_references.txt"
print(f"\nWriting reference data to {output_filename}...")

try:
    with open(output_filename, "w") as f:
        f.write("# Auto-generated reference data for OmniDSP C++ tests\n")
        f.write("# Includes Conv/Corr/Desamp (from SciPy) and CQT (from Librosa)\n\n")

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
                # Flatten the array for writing element by element
                arr_flat = arr.flatten()
            else:
                arr_flat = arr  # Use as is if 1D

            # Write data
            for val in arr_flat:
                if is_complex:
                    # Write real and imaginary parts on separate lines for easier C++ parsing
                    if isinstance(val.real, np.float32):  # Check underlying type
                        f.write(f"{val.real:.9g}f # real\n")
                        f.write(f"{val.imag:.9g}f # imag\n")
                    else:
                        f.write(f"{val.real:.17g} # real\n")
                        f.write(f"{val.imag:.17g} # imag\n")
                elif is_float:
                    # Use 'g' for flexible float format, add 'f' suffix
                    f.write(f"{val:.9g}f\n")
                else:  # Double real
                    # Use 'g' for flexible double format
                    f.write(f"{val:.17g}\n")

            # Write end marker
            f.write(f"# END {name}\n\n")

    print("Finished writing reference data.")

except ImportError:
    print("\nError: Librosa not found. Please ensure it is installed in your environment.")
    print("You can typically install it using: conda install -c conda-forge librosa")
except IOError as e:
    print(f"Error writing file {output_filename}: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

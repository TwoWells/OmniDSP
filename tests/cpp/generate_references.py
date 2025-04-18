# tests/cpp/generate_references.py
# Generates reference data for C++ tests using SciPy and Librosa.
# Includes:
# - SciPy Window functions (float and double)
# - SciPy Convolution/Correlation
# - SciPy FIR Filter + Downsample
# - Librosa CQT
# - SciPy FFT (fft, ifft, rfft, irfft) with different norms

import numpy as np
from scipy import signal, fft as sp_fft  # Use modern scipy.fft
from scipy.signal import windows  # Keep windows import separate for clarity
import librosa
import os
import warnings

# --- Parameters ---
FFT_N = 16  # FFT length, chosen to match fft.cpp tests

# --- Generate Window Coefficients using SciPy ---
print("\nGenerating Window reference data (float and double)...")  # Updated print
window_n = 5
window_kaiser_beta = 8.0

# Generate double precision
hann_coeffs_d = windows.hann(window_n).astype(np.float64)
hamming_coeffs_d = windows.hamming(window_n).astype(np.float64)
kaiser_coeffs_d = windows.kaiser(window_n, beta=window_kaiser_beta).astype(np.float64)
flattop_coeffs_d = windows.flattop(window_n).astype(np.float64)

# Generate float versions by casting doubles (uncommented)
hann_coeffs_f = hann_coeffs_d.astype(np.float32)
hamming_coeffs_f = hamming_coeffs_d.astype(np.float32)
kaiser_coeffs_f = kaiser_coeffs_d.astype(np.float32)
flattop_coeffs_f = flattop_coeffs_d.astype(np.float32)

print(f"  Window N={window_n}, Kaiser Beta={window_kaiser_beta}")


# --- Generate Convolution/Correlation Reference Data ---
print("\nGenerating Convolution/Correlation reference data...")
signal_d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float64)
kernel_d = np.array([0.5, 1.0, 0.5], dtype=np.float64)
kernel_d_asym = np.array([1.0, 2.0, 0.5], dtype=np.float64)
signal_f = signal_d.astype(np.float32)
kernel_f = kernel_d.astype(np.float32)
kernel_f_asym = kernel_d_asym.astype(np.float32)
signal_edge = np.array([1.0, 2.0, 3.0], dtype=np.float64)
kernel_edge = np.array([0.5, 1.0, 0.5], dtype=np.float64)

expected_corr_d = signal.correlate(signal_d, kernel_d, mode="valid")
expected_corr_f = signal.correlate(signal_f, kernel_f, mode="valid")
expected_corr_edge_d = signal.correlate(signal_edge, kernel_edge, mode="valid")
expected_conv_d = signal.convolve(signal_d, kernel_d_asym, mode="valid")
expected_conv_f = signal.convolve(signal_f, kernel_f_asym, mode="valid")
expected_conv_edge_d = signal.convolve(signal_edge, kernel_edge, mode="valid")
print("  Conv/Corr shapes generated.")


# --- Generate Filter + Downsample Reference Data ---
print("\nCalculating Filter+Downsample reference data...")
filter_downsample_sr = 4000.0
filter_downsample_len = 2048
filter_downsample_order = 101
filter_downsample_cutoff = filter_downsample_sr / 4.0
downsample_factor = 2
filter_downsample_freq_low = 500.0
filter_downsample_freq_high = 750.0
filter_downsample_time = np.arange(filter_downsample_len) / filter_downsample_sr
signal_filter_test_d = (
    0.5 * np.sin(2 * np.pi * filter_downsample_freq_low * filter_downsample_time)
    + 0.5 * np.sin(2 * np.pi * filter_downsample_freq_high * filter_downsample_time)
).astype(np.float64)
signal_filter_test_f = signal_filter_test_d.astype(np.float32)
coeffs_d = signal.firwin(
    filter_downsample_order,
    cutoff=filter_downsample_cutoff,
    window="hann",
    fs=filter_downsample_sr,
    scale=True,  # Note: scale=True might differ from C++ default filter design
)
coeffs_f = coeffs_d.astype(np.float32)
filtered_d = signal.correlate(signal_filter_test_d, coeffs_d, mode="valid")
filtered_f = signal.correlate(signal_filter_test_f, coeffs_f, mode="valid")
expected_filter_downsample_d = filtered_d[::downsample_factor]
expected_filter_downsample_f = filtered_f[::downsample_factor]
print(
    f"  Filter+Downsample shapes: D={expected_filter_downsample_d.shape}, F={expected_filter_downsample_f.shape}"
)


# --- Generate CQT Reference Data ---
print("\nCalculating Librosa CQT reference...")
cqt_sr = 44100.0
cqt_hop_length = 512
cqt_fmin = 55.0
cqt_fmax = 1760.0
cqt_bins_per_octave = 12
cqt_n_bins = int(np.ceil(cqt_bins_per_octave * np.log2(cqt_fmax / cqt_fmin)))
cqt_signal_freq = 440.0
cqt_signal_duration = 1.0
cqt_signal_len = int(cqt_signal_duration * cqt_sr)
cqt_time = np.arange(cqt_signal_len) / cqt_sr
cqt_signal_d = np.sin(2 * np.pi * cqt_signal_freq * cqt_time).astype(np.float64)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    librosa_cqt_complex_d = librosa.cqt(
        y=cqt_signal_d,
        sr=cqt_sr,
        hop_length=cqt_hop_length,
        fmin=cqt_fmin,
        n_bins=cqt_n_bins,
        bins_per_octave=cqt_bins_per_octave,
        filter_scale=1,  # Matches default C++? Check C++ implementation if issues arise
        res_type="kaiser_fast",  # Matches default C++? Check C++ implementation if issues arise
        dtype=np.complex128,
    )
# librosa_cqt_complex_f = librosa_cqt_complex_d.astype(np.complex64) # Generate float if needed
print(f"  Librosa CQT calculated. Shape: {librosa_cqt_complex_d.shape}")


# --- Generate FFT Reference Data ---
print(f"\nGenerating FFT reference data (N={FFT_N})...")
# Input Signals
fft_input_real_d = np.arange(FFT_N, dtype=np.float64) / FFT_N  # Simple ramp
fft_input_real_f = fft_input_real_d.astype(np.float32)
fft_input_complex_d = (fft_input_real_d + 1j * fft_input_real_d[::-1]).astype(
    np.complex128
)
fft_input_complex_f = fft_input_complex_d.astype(np.complex64)

# FFT (Complex Forward) Calculations
fft_expected_backward_d = sp_fft.fft(fft_input_complex_d, norm="backward")
fft_expected_backward_f = sp_fft.fft(fft_input_complex_f, norm="backward")
fft_expected_ortho_d = sp_fft.fft(fft_input_complex_d, norm="ortho")
fft_expected_ortho_f = sp_fft.fft(fft_input_complex_f, norm="ortho")
fft_expected_forward_d = sp_fft.fft(fft_input_complex_d, norm="forward")
fft_expected_forward_f = sp_fft.fft(fft_input_complex_f, norm="forward")

# IFFT (Complex Inverse) Calculations
ifft_expected_backward_d = sp_fft.ifft(fft_expected_backward_d, norm="backward")
ifft_expected_backward_f = sp_fft.ifft(fft_expected_backward_f, norm="backward")
ifft_expected_ortho_d = sp_fft.ifft(fft_expected_ortho_d, norm="ortho")
ifft_expected_ortho_f = sp_fft.ifft(fft_expected_ortho_f, norm="ortho")
ifft_expected_forward_d = sp_fft.ifft(fft_expected_forward_d, norm="forward")
ifft_expected_forward_f = sp_fft.ifft(fft_expected_forward_f, norm="forward")

# RFFT (Real Forward) Calculations
rfft_expected_backward_d = sp_fft.rfft(fft_input_real_d, norm="backward")
rfft_expected_backward_f = sp_fft.rfft(fft_input_real_f, norm="backward")
rfft_expected_ortho_d = sp_fft.rfft(fft_input_real_d, norm="ortho")
rfft_expected_ortho_f = sp_fft.rfft(fft_input_real_f, norm="ortho")
rfft_expected_forward_d = sp_fft.rfft(fft_input_real_d, norm="forward")
rfft_expected_forward_f = sp_fft.rfft(fft_input_real_f, norm="forward")

# IRFFT (Real Inverse) Calculations
irfft_expected_backward_d = sp_fft.irfft(
    rfft_expected_backward_d, n=FFT_N, norm="backward"
)
irfft_expected_backward_f = sp_fft.irfft(
    rfft_expected_backward_f, n=FFT_N, norm="backward"
)
irfft_expected_ortho_d = sp_fft.irfft(rfft_expected_ortho_d, n=FFT_N, norm="ortho")
irfft_expected_ortho_f = sp_fft.irfft(rfft_expected_ortho_f, n=FFT_N, norm="ortho")
irfft_expected_forward_d = sp_fft.irfft(
    rfft_expected_forward_d, n=FFT_N, norm="forward"
)
irfft_expected_forward_f = sp_fft.irfft(
    rfft_expected_forward_f, n=FFT_N, norm="forward"
)
print("  FFT references calculated.")


# --- Data Dictionary ---
data_to_write = {
    # Window data (Double)
    "WINDOW_HANN_N5_D": hann_coeffs_d,
    "WINDOW_HAMMING_N5_D": hamming_coeffs_d,
    "WINDOW_KAISER_N5_B8_D": kaiser_coeffs_d,
    "WINDOW_FLATTOP_N5_D": flattop_coeffs_d,
    # Window data (Float) - Added
    "WINDOW_HANN_N5_F": hann_coeffs_f,
    "WINDOW_HAMMING_N5_F": hamming_coeffs_f,
    "WINDOW_KAISER_N5_B8_F": kaiser_coeffs_f,
    "WINDOW_FLATTOP_N5_F": flattop_coeffs_f,
    # Conv/Corr data
    "expected_corr_d": expected_corr_d,
    "expected_corr_f": expected_corr_f,
    "expected_corr_edge_d": expected_corr_edge_d,
    "expected_conv_d": expected_conv_d,
    "expected_conv_f": expected_conv_f,
    "expected_conv_edge_d": expected_conv_edge_d,
    # Filter+Downsample data
    "expected_filter_downsample_d": expected_filter_downsample_d,
    "expected_filter_downsample_f": expected_filter_downsample_f,
    # Librosa CQT data
    "librosa_cqt_complex_d": librosa_cqt_complex_d,
    # FFT Input Data
    "FFT_INPUT_COMPLEX_D": fft_input_complex_d,
    "FFT_INPUT_COMPLEX_F": fft_input_complex_f,
    "FFT_INPUT_REAL_D": fft_input_real_d,
    "FFT_INPUT_REAL_F": fft_input_real_f,
    # FFT Expected Results
    "FFT_EXPECTED_BACKWARD_D": fft_expected_backward_d,
    "FFT_EXPECTED_BACKWARD_F": fft_expected_backward_f,
    "FFT_EXPECTED_ORTHO_D": fft_expected_ortho_d,
    "FFT_EXPECTED_ORTHO_F": fft_expected_ortho_f,
    "FFT_EXPECTED_FORWARD_D": fft_expected_forward_d,
    "FFT_EXPECTED_FORWARD_F": fft_expected_forward_f,
    # IFFT Expected Results - Should ideally match original inputs closely
    "IFFT_EXPECTED_BACKWARD_D": ifft_expected_backward_d,
    "IFFT_EXPECTED_BACKWARD_F": ifft_expected_backward_f,
    "IFFT_EXPECTED_ORTHO_D": ifft_expected_ortho_d,
    "IFFT_EXPECTED_ORTHO_F": ifft_expected_ortho_f,
    "IFFT_EXPECTED_FORWARD_D": ifft_expected_forward_d,
    "IFFT_EXPECTED_FORWARD_F": ifft_expected_forward_f,
    # RFFT Expected Results
    "RFFT_EXPECTED_BACKWARD_D": rfft_expected_backward_d,
    "RFFT_EXPECTED_BACKWARD_F": rfft_expected_backward_f,
    "RFFT_EXPECTED_ORTHO_D": rfft_expected_ortho_d,
    "RFFT_EXPECTED_ORTHO_F": rfft_expected_ortho_f,
    "RFFT_EXPECTED_FORWARD_D": rfft_expected_forward_d,
    "RFFT_EXPECTED_FORWARD_F": rfft_expected_forward_f,
    # IRFFT Expected Results - Should ideally match original inputs closely
    "IRFFT_EXPECTED_BACKWARD_D": irfft_expected_backward_d,
    "IRFFT_EXPECTED_BACKWARD_F": irfft_expected_backward_f,
    "IRFFT_EXPECTED_ORTHO_D": irfft_expected_ortho_d,
    "IRFFT_EXPECTED_ORTHO_F": irfft_expected_ortho_f,
    "IRFFT_EXPECTED_FORWARD_D": irfft_expected_forward_d,
    "IRFFT_EXPECTED_FORWARD_F": irfft_expected_forward_f,
}

# --- Write to File ---
script_dir = os.path.dirname(os.path.abspath(__file__))
output_filename = os.path.join(script_dir, "test_references.txt")
print(f"\nWriting reference data to {output_filename}...")

try:
    with open(output_filename, "w") as f:
        f.write("# Auto-generated reference data for OmniDSP C++ tests\n")
        f.write(
            "# Includes Window(F/D), Conv/Corr, Filter+Downsample, CQT, FFT(F/D)\n\n"  # Updated comment
        )

        for name, arr in data_to_write.items():
            # Determine type for formatting
            is_float = arr.dtype == np.float32 or arr.dtype == np.complex64
            is_complex = np.iscomplexobj(arr)
            is_2d = arr.ndim == 2

            # Write start marker
            f.write(f"# START {name}\n")

            # Add shape information for 2D arrays
            if is_2d:
                f.write(f"# SHAPE: {arr.shape[0]} {arr.shape[1]}\n")
                arr_flat = arr.flatten()  # Flatten for iteration
            else:
                arr_flat = arr

            # Write data
            for val in arr_flat:
                if is_complex:
                    # Use more precision for complex numbers
                    if is_float:  # complex64
                        f.write(f"{val.real:.9g}f # real\n")
                        f.write(f"{val.imag:.9g}f # imag\n")
                    else:  # complex128
                        f.write(f"{val.real:.17g} # real\n")
                        f.write(f"{val.imag:.17g} # imag\n")
                elif is_float:  # float32
                    f.write(f"{val:.9g}f\n")
                else:  # float64 (double)
                    f.write(f"{val:.17g}\n")

            # Write end marker
            f.write(f"# END {name}\n\n")

    print("Finished writing reference data.")

except ImportError as e:
    print(
        f"\nError: Required library not found ({e}). Please ensure SciPy and Librosa are installed."
    )
    print(
        "You can typically install them using: conda install scipy librosa -c conda-forge"
    )
except IOError as e:
    print(f"Error writing file {output_filename}: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

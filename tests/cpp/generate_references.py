# generate_references.py
import numpy as np
from scipy import signal
import os

# --- Define Input Signals and Kernels (same as before) ---
signal_d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float64)
kernel_d = np.array([0.5, 1.0, 0.5], dtype=np.float64)        # Symmetric
kernel_d_asym = np.array([1.0, 2.0, 0.5], dtype=np.float64)   # Asymmetric

signal_f = signal_d.astype(np.float32)
kernel_f = kernel_d.astype(np.float32)
kernel_f_asym = kernel_d_asym.astype(np.float32)

signal_edge = np.array([1.0, 2.0, 3.0], dtype=np.float64)
kernel_edge = np.array([0.5, 1.0, 0.5], dtype=np.float64)

# --- Calculate Expected Results (same as before) ---
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

# --- Data to Write ---
# Store data in a dictionary for easier iteration
data_to_write = {
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
}

# --- Write to File ---
output_filename = "test_references.txt"
print(f"Writing reference data to {output_filename}...")

try:
    with open(output_filename, "w") as f:
        f.write("# Auto-generated reference data for backend_conv_test.cpp\n")
        for name, arr in data_to_write.items():
            # Determine type for formatting
            is_float = arr.dtype == np.float32

            # Write start marker
            f.write(f"# START {name}\n")
            # Write data, one number per line with sufficient precision
            for val in arr:
                if is_float:
                    # Use 'g' for flexible float format, add 'f' suffix
                    f.write(f"{val:.9g}f\n")
                else:
                    # Use 'g' for flexible double format
                    f.write(f"{val:.17g}\n")
            # Write end marker
            f.write(f"# END {name}\n\n")
    print("Finished writing reference data.")

except IOError as e:
    print(f"Error writing file {output_filename}: {e}")

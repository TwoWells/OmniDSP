# omnidsp_py/api.py

"""
Python wrappers and factory functions for the OmniDSP library.

This module provides a user-friendly Python interface that dispatches
calls to the appropriate C++ bindings based on NumPy data types or
explicit precision requests. It defines convenience functions (like fft, rfft)
and factory functions (like create_fft_plan, create_cqt_plan) for creating
plan objects.
"""

import numpy as np
# Import standard typing utilities
from typing import Callable, Union, Optional, TypeVar, cast
import warnings  # Import warnings module (currently unused, but good practice)

# --- Import C++ components ---
# Import the compiled C++ extension module, aliased as 'cpp' for clarity
from . import omnidsp_py as cpp
# Import Enums and specific Plan classes which are needed directly by the factory functions
from .omnidsp_py import (
    Precision, Direction, Domain, NormMode,  # Enums
    FFTPlanFloat, FFTPlanDouble,             # Specific FFT Plan classes
    CQTPlanFloat, CQTPlanDouble              # Specific CQT Plan classes
    # The C++ bound Window class is accessed via cpp.Window
    # The C++ bound fft, ifft, etc functions are accessed via cpp.fft, cpp.ifft
)

# --- Type Aliases ---
# Define type hints for NumPy arrays for better code readability and static analysis.
# Using generic np.ndarray for broader compatibility, but implies float32
ArrayF32 = np.ndarray
ArrayF64 = np.ndarray  # Implies float64
ArrayC64 = np.ndarray  # Implies complex64
ArrayC128 = np.ndarray  # Implies complex128
# Type hint for real-valued NumPy arrays
ArrayReal = Union[ArrayF32, ArrayF64]
# Type hint for complex-valued NumPy arrays
ArrayComplex = Union[ArrayC64, ArrayC128]
# Type hint for either FFT plan type
AnyFFTPlan = Union[FFTPlanFloat, FFTPlanDouble]
# Type hint for either CQT plan type
AnyCQTPlan = Union[CQTPlanFloat, CQTPlanDouble]

# Type variable for hinting the signature of the window function callable.
# It takes an array and returns an array.
T_WindowFunc = TypeVar(
    'T_WindowFunc', bound=Callable[[np.ndarray], np.ndarray])

# Default values matching C++ implementation (from bindings.cpp/cqt.h)
DEFAULT_SPARSITY_THRESHOLD = 1e-5
DEFAULT_FIR_FILTER_ORDER = 101

# --- Wrapper Functions for FFT Convenience Functions ---
# These Python functions provide a cleaner interface to the overloaded C++ convenience
# functions bound in bindings.cpp (accessed via the 'cpp' alias). They perform basic
# type checking before calling the C++ function, which handles the actual dispatch.


def fft(input_array: ArrayComplex) -> ArrayComplex:
    """
    Performs Complex-to-Complex forward FFT (NormMode.BACKWARD).

    Dispatches to the appropriate C++ float32 or float64 implementation
    based on the input NumPy array's dtype.

    Args:
        input_array: 1D NumPy array (complex64 or complex128).

    Returns:
        1D NumPy array of the FFT result, with the same dtype as the input.

    Raises:
        TypeError: If input dtype is not complex64 or complex128.
        RuntimeError: If the C++ binding encounters an error (e.g., input not 1D).
    """
    # Check if the input array's data type is supported.
    if not (input_array.dtype == np.complex64 or input_array.dtype == np.complex128):
        raise TypeError(
            f"Unsupported dtype {input_array.dtype} for fft. Expected complex64 or complex128.")
    # Call the bound C++ function (cpp.fft), which is overloaded in bindings.cpp.
    # pybind11 automatically selects the correct C++ overload based on the input array type.
    return cpp.fft(input_array)


def ifft(input_array: ArrayComplex) -> ArrayComplex:
    """
    Performs Complex-to-Complex inverse FFT (NormMode.BACKWARD).

    Dispatches to the appropriate C++ float32 or float64 implementation
    based on the input NumPy array's dtype.

    Args:
        input_array: 1D NumPy array (complex64 or complex128).

    Returns:
        1D NumPy array of the IFFT result, with the same dtype as the input.

    Raises:
        TypeError: If input dtype is not complex64 or complex128.
        RuntimeError: If the C++ binding encounters an error.
    """
    # Check input dtype.
    if not (input_array.dtype == np.complex64 or input_array.dtype == np.complex128):
        raise TypeError(
            f"Unsupported dtype {input_array.dtype} for ifft. Expected complex64 or complex128.")
    # Call the bound C++ function (cpp.ifft).
    return cpp.ifft(input_array)


def rfft(input_array: ArrayReal) -> ArrayComplex:
    """
    Performs Real-to-Complex forward FFT (NormMode.BACKWARD).

    Dispatches to the appropriate C++ float32 or float64 implementation
    based on the input NumPy array's dtype.
    Returns N/2 + 1 complex frequency points.

    Args:
        input_array: 1D NumPy array (float32 or float64).

    Returns:
        1D NumPy array (complex64 or complex128) of size N/2 + 1.

    Raises:
        TypeError: If input dtype is not float32 or float64.
        RuntimeError: If the C++ binding encounters an error.
    """
    # Check input dtype.
    if not (input_array.dtype == np.float32 or input_array.dtype == np.float64):
        raise TypeError(
            f"Unsupported dtype {input_array.dtype} for rfft. Expected float32 or float64.")
     # Call the bound C++ function (cpp.rfft).
    return cpp.rfft(input_array)


def irfft(input_array: ArrayComplex) -> ArrayReal:
    """
    Performs Complex-to-Real inverse FFT (NormMode.BACKWARD).

    Dispatches to the appropriate C++ float32 or float64 implementation
    based on the input NumPy array's dtype.
    Input spectrum must have Hermitian symmetry for the output to be purely real.
    Returns N real points, where N = 2*(Nc-1) if input length Nc > 1, or N=1 if Nc=1.

    Args:
        input_array: 1D NumPy array (complex64 or complex128) of size Nc = N/2 + 1,
                     possessing Hermitian symmetry.

    Returns:
        1D NumPy array (float32 or float64) of size N.

    Raises:
        TypeError: If input dtype is not complex64 or complex128.
        RuntimeError: If the C++ binding encounters an error.
    """
    # Check input dtype.
    if not (input_array.dtype == np.complex64 or input_array.dtype == np.complex128):
        raise TypeError(
            f"Unsupported dtype {input_array.dtype} for irfft. Expected complex64 or complex128.")
     # Call the bound C++ function (cpp.irfft).
    return cpp.irfft(input_array)


# --- Factory Functions for Creating Plan Objects ---
# These functions provide a Pythonic way to create instances of the C++ plan objects
# (FFTPlanFloat, FFTPlanDouble, CQTPlanFloat, CQTPlanDouble) bound in bindings.cpp.

def create_fft_plan(
    length: int,
    precision: Precision,
    direction: Direction,
    domain: Domain,
    norm: NormMode = NormMode.BACKWARD  # Default normalization mode
) -> AnyFFTPlan:
    """
    Factory function to create an FFTPlan instance.

    Chooses between FFTPlanFloat and FFTPlanDouble based on the precision argument.

    Args:
        length: The size 'N' of the transform (number of points).
        precision: The desired precision (omnidsp_py.Precision.SINGLE or
                   omnidsp_py.Precision.DOUBLE).
        direction: Transform direction (omnidsp_py.Direction.FORWARD or
                   omnidsp_py.Direction.INVERSE).
        domain: Transform domain (omnidsp_py.Domain.COMPLEX or
                omnidsp_py.Domain.REAL).
        norm: Normalization mode (default: omnidsp_py.NormMode.BACKWARD).

    Returns:
        An instance of FFTPlanFloat or FFTPlanDouble.

    Raises:
        ValueError: If an unsupported precision enum value is provided.
        RuntimeError: If C++ plan creation fails (e.g., invalid length for backend).
    """
    # Instantiate the appropriate C++ plan class based on the requested precision.
    if precision == Precision.SINGLE:
        # Call the constructor bound in bindings.cpp for FFTPlanFloat
        return FFTPlanFloat(length, precision, direction, domain, norm)
    elif precision == Precision.DOUBLE:
        # Call the constructor bound in bindings.cpp for FFTPlanDouble
        return FFTPlanDouble(length, precision, direction, domain, norm)
    else:
        # Handle invalid precision enum values.
        prec_repr = repr(precision) if isinstance(
            precision, Precision) else str(precision)
        raise ValueError(f"Unsupported precision: {prec_repr}")


def create_cqt_plan(
    sample_rate: float,
    hop_length: int,            # Added hop_length parameter
    lowest_freq: float,
    highest_freq: float,
    bins_per_octave: int,
    window_function: T_WindowFunc,  # Use TypeVar for better hinting of callable type
    precision: Precision = Precision.DOUBLE,  # Default precision to DOUBLE
    sparsity_threshold: Optional[float] = None,  # Optional sparsity threshold
    fir_filter_order: Optional[int] = None    # Optional FIR filter order
) -> AnyCQTPlan:
    """
    Factory function to create a recursive CQTPlan instance.

    Chooses between CQTPlanFloat and CQTPlanDouble based on the precision argument.

    Args:
        sample_rate: Sample rate of the input signal in Hz.
        hop_length: The number of samples between consecutive CQT frames. Must be
                    divisible by 2^(num_octaves - 1).
        lowest_freq: Lowest frequency of interest in Hz (> 0).
        highest_freq: Highest frequency of interest in Hz (<= sample_rate / 2).
        bins_per_octave: Number of CQT bins per octave (> 0).
        window_function: A Python callable (function or lambda). It will be called
                         internally by the C++ backend during kernel generation.
                         The callable will receive a dummy NumPy array argument whose
                         size indicates the required window length for a specific CQT bin.
                         The callable MUST return a 1D NumPy array of window coefficients
                         of that exact length and with the correct dtype (float32 for
                         SINGLE precision plan, float64 for DOUBLE precision plan).
                         Example: `lambda arr: np.hanning(len(arr)).astype(arr.dtype)`
        precision: The desired precision (omnidsp_py.Precision.SINGLE or
                   omnidsp_py.Precision.DOUBLE). Defaults to DOUBLE.
        sparsity_threshold: Threshold below which kernel values are treated as zero
                            during precomputation. If None, uses C++ default (1e-5).
        fir_filter_order: Order (length) of the FIR anti-aliasing filter used in
                          recursive downsampling. Must be odd. If None, uses C++
                          default (101).

    Returns:
        An instance of CQTPlanFloat or CQTPlanDouble.

    Raises:
        ValueError: If an unsupported precision enum value is provided.
        TypeError: If window_function is not callable.
        RuntimeError: If C++ plan creation fails (e.g., invalid args, hop_length issue,
                      window function error during kernel generation).
    """
    # Validate that the window_function is actually callable.
    if not callable(window_function):
        raise TypeError(
            "window_function must be a callable (e.g., a Python function or lambda).")

    # Determine the sparsity threshold and FIR filter order to pass to C++,
    # using the C++ defaults if the Python arguments are None.
    # The C++ binding layer expects these values explicitly.
    sparsity_thresh_cpp = sparsity_threshold if sparsity_threshold is not None else DEFAULT_SPARSITY_THRESHOLD
    fir_order_cpp = fir_filter_order if fir_filter_order is not None else DEFAULT_FIR_FILTER_ORDER

    # Instantiate the appropriate C++ plan class based on the requested precision.
    if precision == Precision.SINGLE:
        # Call the constructor bound in bindings.cpp for CQTPlanFloat.
        # The Python callable `window_function` is passed directly; the C++ binding lambda handles the adaptation.
        return CQTPlanFloat(
            sample_rate,
            hop_length,
            lowest_freq,
            highest_freq,
            bins_per_octave,
            window_function,  # Pass the Python callable
            float(sparsity_thresh_cpp),  # Ensure correct type for float plan
            fir_order_cpp
        )
    elif precision == Precision.DOUBLE:
        # Call the constructor bound in bindings.cpp for CQTPlanDouble.
        return CQTPlanDouble(
            sample_rate,
            hop_length,
            lowest_freq,
            highest_freq,
            bins_per_octave,
            window_function,  # Pass the Python callable
            float(sparsity_thresh_cpp),  # Ensure correct type for double plan
            fir_order_cpp
        )
    else:
        # Handle invalid precision enum values.
        prec_repr = repr(precision) if isinstance(
            precision, Precision) else str(precision)
        raise ValueError(f"Unsupported precision: {prec_repr}")

# Note: Python wrappers are generally not needed for the Window class static methods
# (like Window.hann). The __init__.py file can directly expose the C++ bound
# Window class from the 'cpp' module. The overloaded static methods defined
# in bindings.cpp will handle the type dispatching automatically when called
# like omnidsp_py.Window.hann(numpy_array).

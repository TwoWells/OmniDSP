# omnidsp_py/api.py

"""
Python wrappers and factory functions for the OmniDSP library.

This module provides a user-friendly Python interface that dispatches
calls to the appropriate C++ bindings based on NumPy data types or
explicit precision requests.
"""

import numpy as np
from typing import Callable, Union, Sequence, TypeVar, cast
import warnings # Import warnings

# --- Import C++ components ---
from . import omnidsp_py as cpp # Alias for the compiled C++ module
# Import Enums and Plan classes which are needed directly
from .omnidsp_py import (
    Precision, Direction, Domain, NormMode,
    FFTPlanFloat, FFTPlanDouble,
    CQTPlanFloat, CQTPlanDouble
    # The C++ bound Window class is accessed via cpp.Window
    # The C++ bound fft,ifft,etc functions are accessed via cpp.fft, cpp.ifft
)

# Type aliases for NumPy arrays and Plans
ArrayF32 = np.ndarray[np.float32, np.dtype[np.float32]]
ArrayF64 = np.ndarray[np.float64, np.dtype[np.float64]]
ArrayC64 = np.ndarray[np.complex64, np.dtype[np.complex64]]
ArrayC128 = np.ndarray[np.complex128, np.dtype[np.complex128]]
ArrayReal = Union[ArrayF32, ArrayF64]
ArrayComplex = Union[ArrayC64, ArrayC128]
AnyFFTPlan = Union[FFTPlanFloat, FFTPlanDouble]
AnyCQTPlan = Union[CQTPlanFloat, CQTPlanDouble]

# Type variable for Window function callable hint
T_WindowFunc = TypeVar('T_WindowFunc', bound=Callable[[np.ndarray], np.ndarray])


# --- Wrapper Functions for FFT ---
# These call the overloaded C++ functions aliased via 'cpp'

def fft(input_array: ArrayComplex) -> ArrayComplex:
    """
    Performs Complex-to-Complex forward FFT (NormMode.BACKWARD).

    Dispatches to float32 or float64 implementation based on input dtype.

    Args:
        input_array: 1D NumPy array (complex64 or complex128).

    Returns:
        1D NumPy array of the same dtype as input.

    Raises:
        TypeError: If input dtype is not complex64 or complex128.
        RuntimeError: If input is not 1D (from C++ binding).
    """
    # Type check input dtype
    if not (input_array.dtype == np.complex64 or input_array.dtype == np.complex128):
        raise TypeError(f"Unsupported dtype {input_array.dtype} for fft. Expected complex64 or complex128.")
    # Call the C++ overloaded function via the alias
    return cpp.fft(input_array)

def ifft(input_array: ArrayComplex) -> ArrayComplex:
    """
    Performs Complex-to-Complex inverse FFT (NormMode.BACKWARD).

    Dispatches to float32 or float64 implementation based on input dtype.

    Args:
        input_array: 1D NumPy array (complex64 or complex128).

    Returns:
        1D NumPy array of the same dtype as input.

    Raises:
        TypeError: If input dtype is not complex64 or complex128.
        RuntimeError: If input is not 1D (from C++ binding).
    """
    # Type check input dtype
    if not (input_array.dtype == np.complex64 or input_array.dtype == np.complex128):
        raise TypeError(f"Unsupported dtype {input_array.dtype} for ifft. Expected complex64 or complex128.")
    # Call the C++ overloaded function via the alias
    return cpp.ifft(input_array)

def rfft(input_array: ArrayReal) -> ArrayComplex:
    """
    Performs Real-to-Complex forward FFT (NormMode.BACKWARD).

    Dispatches to float32 or float64 implementation based on input dtype.
    Returns N/2 + 1 complex points.

    Args:
        input_array: 1D NumPy array (float32 or float64).

    Returns:
        1D NumPy array (complex64 or complex128) of size N/2 + 1.

    Raises:
        TypeError: If input dtype is not float32 or float64.
        RuntimeError: If input is not 1D (from C++ binding).
    """
     # Type check input dtype
    if not (input_array.dtype == np.float32 or input_array.dtype == np.float64):
        raise TypeError(f"Unsupported dtype {input_array.dtype} for rfft. Expected float32 or float64.")
     # Call the C++ overloaded function via the alias
    return cpp.rfft(input_array)

def irfft(input_array: ArrayComplex) -> ArrayReal:
    """
    Performs Complex-to-Real inverse FFT (NormMode.BACKWARD).

    Dispatches to float32 or float64 implementation based on input dtype.
    Input must have Hermitian symmetry for the output to be purely real.
    Returns N real points, where N = 2*(Nc-1) if Nc > 1, or N=1 if Nc=1.

    Args:
        input_array: 1D NumPy array (complex64 or complex128) of size Nc = N/2 + 1.

    Returns:
        1D NumPy array (float32 or float64) of size N.

    Raises:
        TypeError: If input dtype is not complex64 or complex128.
        RuntimeError: If input is not 1D (from C++ binding).
    """
    # Type check input dtype
    if not (input_array.dtype == np.complex64 or input_array.dtype == np.complex128):
        raise TypeError(f"Unsupported dtype {input_array.dtype} for irfft. Expected complex64 or complex128.")
     # Call the C++ overloaded function via the alias
    return cpp.irfft(input_array)


# --- Factory Functions for Plans ---

def create_fft_plan(
    length: int,
    precision: Precision,
    direction: Direction,
    domain: Domain,
    norm: NormMode = NormMode.BACKWARD
) -> AnyFFTPlan:
    """
    Factory function to create an FFTPlan instance.

    Args:
        length: The size 'N' of the transform.
        precision: The desired precision (Precision.SINGLE or Precision.DOUBLE).
        direction: Transform direction (Direction.FORWARD or Direction.INVERSE).
        domain: Transform domain (Domain.COMPLEX or Domain.REAL).
        norm: Normalization mode (default: NormMode.BACKWARD).

    Returns:
        An instance of FFTPlanFloat or FFTPlanDouble.

    Raises:
        ValueError: If an unsupported precision is provided.
        RuntimeError: If C++ plan creation fails (e.g., invalid length).
    """
    if precision == Precision.SINGLE:
        return FFTPlanFloat(length, precision, direction, domain, norm)
    elif precision == Precision.DOUBLE:
        return FFTPlanDouble(length, precision, direction, domain, norm)
    else:
        # Ensure Precision is an enum before formatting
        prec_repr = repr(precision) if isinstance(precision, Precision) else str(precision)
        raise ValueError(f"Unsupported precision: {prec_repr}")

def create_cqt_plan(
    sample_rate: float,
    lowest_freq: float,
    highest_freq: float,
    bins_per_octave: int,
    window_function: T_WindowFunc, # Use TypeVar for better hinting
    precision: Precision = Precision.DOUBLE
) -> AnyCQTPlan:
    """
    Factory function to create a CQTPlan instance.

    Args:
        sample_rate: Sample rate in Hz.
        lowest_freq: Lowest frequency of interest in Hz (> 0).
        highest_freq: Highest frequency of interest in Hz (<= sample_rate / 2).
        bins_per_octave: Number of bins per octave (> 0).
        window_function: A Python callable. It will be called internally by C++
                         with a dummy NumPy array argument whose size indicates
                         the required window length. The callable should return
                         a 1D NumPy array of window coefficients of that length
                         and the correct dtype (float32 for SINGLE precision plan,
                         float64 for DOUBLE precision plan).
                         Example: `lambda arr: np.hanning(len(arr)).astype(arr.dtype)`
        precision: The desired precision (Precision.SINGLE or Precision.DOUBLE).
                   Defaults to DOUBLE.

    Returns:
        An instance of CQTPlanFloat or CQTPlanDouble.

    Raises:
        ValueError: If an unsupported precision is provided.
        TypeError: If window_function is not callable, or if it returns
                   an incompatible type/dtype during C++ execution.
        RuntimeError: If C++ plan creation fails (e.g., invalid args, window func error).
    """
    if not callable(window_function):
        raise TypeError("window_function must be a callable (e.g., a Python function).")

    # Pass the Python callable directly. The C++ binding lambda handles adaptation.
    if precision == Precision.SINGLE:
        # Warn if the provided callable looks like it might return the wrong dtype? Optional.
        # warnings.warn("Ensure window_function returns float32 NumPy array for SINGLE precision CQTPlan")
        return CQTPlanFloat(sample_rate, lowest_freq, highest_freq, bins_per_octave, window_function)
    elif precision == Precision.DOUBLE:
        # warnings.warn("Ensure window_function returns float64 NumPy array for DOUBLE precision CQTPlan")
        return CQTPlanDouble(sample_rate, lowest_freq, highest_freq, bins_per_octave, window_function)
    else:
        prec_repr = repr(precision) if isinstance(precision, Precision) else str(precision)
        raise ValueError(f"Unsupported precision: {prec_repr}")

# Note: No Python wrappers needed for Window static methods, as __init__.py
# exposes the C++ bound Window class (via the cpp alias internally if needed,
# but publicly as omnidsp_py.Window), which already contains the overloaded methods.
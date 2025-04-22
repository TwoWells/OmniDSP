# -*- coding: utf-8 -*-
"""
High-level Python API for the OmniDSP library.

This module provides Python functions that wrap the underlying C++ bindings,
offering a more convenient interface using NumPy arrays and standard Python types.
"""

import numpy as np
import warnings
from typing import Optional, Union, Sequence, Type

# Attempt to import the compiled C++ bindings module
try:
    from ._omnidsp_cpp import (
        # Core Classes/Enums
        OmniDSP,
        Backend,
        Status,  # noqa: F401 - Part of public API, potentially used by callers
        DataType,  # noqa: F401 - Part of public API
        Domain,  # noqa: F401 - Part of public API
        WindowType,
        ConvolutionMode,
        WindowSpecFloat,  # noqa: F401 - Part of public API
        WindowSpecDouble,  # noqa: F401 - Part of public API
        OmniDSPError,  # Custom exception class
        # Plan Classes (Example Float versions, add Double/Complex as needed)
        FFTPlanFloatComplex,
        RFFTPlanFloat,
        CQTPlanFloat,
        ConvolutionPlanFloat,
        CorrelationPlanFloat,
        ResamplePlanFloat,
        # Add Double/Complex plan types...
        FFTPlanDoubleComplex,
        RFFTPlanDouble,
        CQTPlanDouble,
        # ConvolutionPlanDouble, ConvolutionPlanFloatComplex, ConvolutionPlanDoubleComplex, # TODO: Bind these
        # CorrelationPlanDouble, CorrelationPlanFloatComplex, CorrelationPlanDoubleComplex, # TODO: Bind these
        ResamplePlanDouble,
    )

    _bindings_available = True
except ImportError:
    warnings.warn(
        "OmniDSP C++ bindings (_omnidsp_cpp) not found. API functions will not work.",
        ImportWarning,
    )
    _bindings_available = False
    # Define dummy types/enums if bindings are missing to allow script import?
    # Or let it fail later when functions are called. Let it fail for now.


# Type hint aliases
ArrayLike = Union[np.ndarray, Sequence[Union[float, complex]]]
Plan = Union[
    FFTPlanFloatComplex,
    FFTPlanDoubleComplex,
    RFFTPlanFloat,
    RFFTPlanDouble,
    CQTPlanFloat,
    CQTPlanDouble,
    ConvolutionPlanFloat,  # Add others when bound
    CorrelationPlanFloat,  # Add others when bound
    ResamplePlanFloat,
    ResamplePlanDouble,
]

# --- Helper Functions ---


def _get_dsp_instance(
    dsp_instance: Optional[OmniDSP] = None, backend: Backend = Backend.Stub
) -> OmniDSP:
    """Gets or creates an OmniDSP instance."""
    if not _bindings_available:
        raise RuntimeError("OmniDSP C++ bindings are not available.")
    if dsp_instance is not None:
        if not isinstance(dsp_instance, OmniDSP):
            raise TypeError("dsp_instance must be an OmniDSP object.")
        return dsp_instance
    else:
        # Create a new instance (caller is responsible for its lifetime if needed beyond the function call)
        return OmniDSP.create(backend)


def _ensure_numpy(arr: ArrayLike, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Ensure input is a NumPy array, optionally converting dtype."""
    if not isinstance(arr, np.ndarray):
        # Attempt conversion, letting NumPy handle potential errors
        try:
            arr = np.asarray(arr, dtype=dtype)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Input could not be converted to a NumPy array: {e}"
            ) from e
    elif dtype is not None and arr.dtype != dtype:
        # Ensure conversion doesn't lose precision unexpectedly if casting down
        if np.issubdtype(dtype, np.integer) and not np.issubdtype(
            arr.dtype, np.integer
        ):
            warnings.warn(
                f"Casting non-integer array to {dtype}, potential precision loss.",
                RuntimeWarning,
            )
        elif np.issubdtype(dtype, np.floating) and np.issubdtype(
            arr.dtype, np.complexfloating
        ):
            warnings.warn(
                f"Casting complex array to {dtype}, discarding imaginary part.",
                RuntimeWarning,
            )
            arr = arr.real  # Explicitly take real part before casting
        try:
            arr = arr.astype(dtype, casting="safe")  # Use 'safe' casting by default
        except TypeError as e:
            # Allow unsafe casting if explicitly requested or handle common cases?
            # For now, re-raise with a more informative message.
            raise TypeError(
                f"Cannot safely cast array from {arr.dtype} to {dtype}: {e}"
            ) from e

    if arr.ndim != 1:
        raise ValueError(
            f"Input must be a 1D array or sequence, but got {arr.ndim} dimensions."
        )
    # Ensure C-contiguity for compatibility with C++ std::span/vector expectations
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def _get_expected_plan_type(plan_type_name: str, dtype: np.dtype) -> type:
    """Helper to get the correct Plan class based on name and dtype."""
    if dtype == np.float32:
        suffix = "Float"
    elif dtype == np.float64:
        suffix = "Double"
    elif dtype == np.complex64:
        suffix = "FloatComplex"
    elif dtype == np.complex128:
        suffix = "DoubleComplex"
    else:
        raise TypeError(f"Unsupported dtype for Plan: {dtype}")

    full_class_name = f"{plan_type_name}{suffix}"
    plan_class = globals().get(full_class_name)  # Access class from module globals
    if plan_class is None:
        # This check might be redundant if bindings cover all types, but good practice
        raise TypeError(
            f"Plan type '{full_class_name}' not available in bindings for dtype {dtype}."
        )
    return plan_class


# --- Main API Functions ---


def create_dsp_instance(backend: Backend = Backend.Stub) -> OmniDSP:
    """
    Creates and returns an OmniDSP instance configured for the specified backend.

    Args:
        backend (Backend): The desired computation backend (Stub, Accelerate, OneMKL).
                           Defaults to Stub.

    Returns:
        OmniDSP: The created OmniDSP instance.

    Raises:
        OmniDSPError: If the instance creation fails (e.g., backend not supported).
        RuntimeError: If C++ bindings are not available.
    """
    if not _bindings_available:
        raise RuntimeError("OmniDSP C++ bindings are not available.")
    # The C++ factory handles error checking via OmniExpected,
    # which the binding translates to OmniDSPError exception.
    return OmniDSP.create(backend)


# --- DSP Operations ---


def convolve(
    input_arr: ArrayLike,
    kernel_arr: ArrayLike,
    mode: ConvolutionMode = ConvolutionMode.Full,
    dsp_instance: Optional[OmniDSP] = None,
) -> np.ndarray:
    """
    Performs 1D convolution.

    Args:
        input_arr: The input signal (1D NumPy array or sequence).
        kernel_arr: The kernel (1D NumPy array or sequence). Must have the same dtype.
        mode: The convolution mode (Full, Same, Valid). Defaults to Full.
        dsp_instance: Optional pre-existing OmniDSP instance. If None, a temporary
                      Stub instance is used.

    Returns:
        np.ndarray: The convolution result with the same dtype as input.

    Raises:
        OmniDSPError: If the C++ library encounters an error.
        TypeError: If input types are incompatible or unsupported.
        ValueError: If inputs are not 1D.
        RuntimeError: If C++ bindings are not available or other unexpected errors occur.
    """
    dsp = _get_dsp_instance(dsp_instance)  # Gets or creates instance
    input_np = _ensure_numpy(input_arr)
    kernel_np = _ensure_numpy(
        kernel_arr, dtype=input_np.dtype
    )  # Ensure kernel matches input dtype

    # The C++ binding for OmniDSP::convolve handles the vector conversion and type dispatch
    try:
        # Assumes the pybind11 bindings for convolve accept numpy arrays directly
        result_vec_or_arr = dsp.convolve(input_np, kernel_np, mode)
        # If the binding returns a std::vector, convert it back
        if isinstance(result_vec_or_arr, list):  # pybind11 default for vector
            return np.array(result_vec_or_arr, dtype=input_np.dtype)
        elif isinstance(
            result_vec_or_arr, np.ndarray
        ):  # If binding returns numpy directly
            return result_vec_or_arr
        else:
            raise TypeError(
                f"Unexpected return type from C++ convolve binding: {type(result_vec_or_arr)}"
            )
    except OmniDSPError as e:
        # Re-raise the specific error from C++
        raise e
    except Exception as e:
        # Catch other potential errors (e.g., pybind type conversion)
        raise RuntimeError(f"Convolution failed: {e}") from e


def correlate(
    input_arr: ArrayLike,
    template_arr: ArrayLike,
    mode: ConvolutionMode = ConvolutionMode.Full,
    dsp_instance: Optional[OmniDSP] = None,
) -> np.ndarray:
    """
    Performs 1D cross-correlation.

    Args:
        input_arr: The input signal (1D NumPy array or sequence).
        template_arr: The template signal (1D NumPy array or sequence). Must have the same dtype.
        mode: The correlation mode (Full, Same, Valid). Defaults to Full.
        dsp_instance: Optional pre-existing OmniDSP instance. If None, a temporary
                      Stub instance is used.

    Returns:
        np.ndarray: The cross-correlation result with the same dtype as input.

    Raises:
        OmniDSPError: If the C++ library encounters an error.
        TypeError: If input types are incompatible or unsupported.
        ValueError: If inputs are not 1D.
        RuntimeError: If C++ bindings are not available or other unexpected errors occur.
    """
    dsp = _get_dsp_instance(dsp_instance)
    input_np = _ensure_numpy(input_arr)
    template_np = _ensure_numpy(template_arr, dtype=input_np.dtype)

    try:
        result_vec_or_arr = dsp.correlate(input_np, template_np, mode)
        if isinstance(result_vec_or_arr, list):
            return np.array(result_vec_or_arr, dtype=input_np.dtype)
        elif isinstance(result_vec_or_arr, np.ndarray):
            return result_vec_or_arr
        else:
            raise TypeError(
                f"Unexpected return type from C++ correlate binding: {type(result_vec_or_arr)}"
            )
    except OmniDSPError as e:
        raise e
    except Exception as e:
        raise RuntimeError(f"Correlation failed: {e}") from e


# --- One-off FFTs ---


def fft(input_arr: ArrayLike, dsp_instance: Optional[OmniDSP] = None) -> np.ndarray:
    """Computes the 1D complex Fast Fourier Transform (FFT)."""
    dsp = _get_dsp_instance(dsp_instance)
    # Ensure input is complex, default to complex128 if real input provided
    input_np = _ensure_numpy(input_arr)
    if not np.iscomplexobj(input_np):
        input_np = input_np.astype(np.complex128)
    # Ensure contiguity after potential type change
    input_np = np.ascontiguousarray(input_np)

    try:
        result_vec_or_arr = dsp.fft(input_np)  # Assumes binding handles dtype dispatch
        if isinstance(result_vec_or_arr, list):
            return np.array(result_vec_or_arr, dtype=input_np.dtype)
        elif isinstance(result_vec_or_arr, np.ndarray):
            return result_vec_or_arr
        else:
            raise TypeError(
                f"Unexpected return type from C++ fft binding: {type(result_vec_or_arr)}"
            )
    except OmniDSPError as e:
        raise e
    except Exception as e:
        raise RuntimeError(f"FFT failed: {e}") from e


def ifft(input_arr: ArrayLike, dsp_instance: Optional[OmniDSP] = None) -> np.ndarray:
    """Computes the 1D inverse complex FFT (scaled by 1/N)."""
    dsp = _get_dsp_instance(dsp_instance)
    # Ensure input is complex, matching precision if possible
    dtype_in = (
        np.complex64 if np.result_type(input_arr) == np.complex64 else np.complex128
    )
    input_np = _ensure_numpy(input_arr, dtype=dtype_in)
    if not np.iscomplexobj(input_np):
        raise TypeError("Input for ifft must be complex.")

    try:
        result_vec_or_arr = dsp.ifft(input_np)  # Assumes binding handles dtype dispatch
        # Note: C++ binding applies 1/N scaling for the one-off ifft
        if isinstance(result_vec_or_arr, list):
            return np.array(result_vec_or_arr, dtype=input_np.dtype)
        elif isinstance(result_vec_or_arr, np.ndarray):
            return result_vec_or_arr
        else:
            raise TypeError(
                f"Unexpected return type from C++ ifft binding: {type(result_vec_or_arr)}"
            )
    except OmniDSPError as e:
        raise e
    except Exception as e:
        raise RuntimeError(f"IFFT failed: {e}") from e


def rfft(input_arr: ArrayLike, dsp_instance: Optional[OmniDSP] = None) -> np.ndarray:
    """Computes the 1D real-to-complex FFT."""
    dsp = _get_dsp_instance(dsp_instance)
    # Ensure input is real, matching precision if possible
    dtype_in = np.float32 if np.result_type(input_arr) == np.float32 else np.float64
    input_np = _ensure_numpy(input_arr, dtype=dtype_in)
    if np.iscomplexobj(input_np):
        raise TypeError("Input for rfft must be real.")

    output_dtype = np.complex64 if input_np.dtype == np.float32 else np.complex128

    try:
        result_vec_or_arr = dsp.rfft(input_np)  # Assumes binding handles dtype dispatch
        if isinstance(result_vec_or_arr, list):
            return np.array(result_vec_or_arr, dtype=output_dtype)
        elif isinstance(result_vec_or_arr, np.ndarray):
            return result_vec_or_arr.astype(
                output_dtype, copy=False
            )  # Ensure correct complex type
        else:
            raise TypeError(
                f"Unexpected return type from C++ rfft binding: {type(result_vec_or_arr)}"
            )
    except OmniDSPError as e:
        raise e
    except Exception as e:
        raise RuntimeError(f"RFFT failed: {e}") from e


def irfft(
    input_arr: ArrayLike,
    n: Optional[int] = None,
    dsp_instance: Optional[OmniDSP] = None,
) -> np.ndarray:
    """Computes the 1D inverse real FFT (complex-to-real, scaled by 1/N)."""
    dsp = _get_dsp_instance(dsp_instance)
    # Ensure input is complex
    dtype_in = (
        np.complex64 if np.result_type(input_arr) == np.complex64 else np.complex128
    )
    input_np = _ensure_numpy(input_arr, dtype=dtype_in)
    if not np.iscomplexobj(input_np):
        raise TypeError("Input for irfft must be complex.")

    if n is None:
        # Infer output length N from input length N/2 + 1
        if input_np.shape[0] < 1:
            n = 0  # Handle empty input case
        else:
            n = (input_np.shape[0] - 1) * 2
            if n == 0 and input_np.shape[0] == 1:
                n = 1  # Handle N=1 case (DC only input)
    elif not isinstance(n, int) or n < 0:
        raise ValueError("Output length n must be a non-negative integer.")
    # Basic validation: input size must correspond roughly to n
    if n > 0 and input_np.shape[0] != (n // 2) + 1:
        warnings.warn(
            f"Input array size {input_np.shape[0]} does not match expected size {(n//2)+1} for output length {n}.",
            RuntimeWarning,
        )

    output_dtype = np.float32 if input_np.dtype == np.complex64 else np.float64

    try:
        result_vec_or_arr = dsp.irfft(
            input_np, n
        )  # Assumes binding handles dtype dispatch
        # Note: C++ binding applies 1/N scaling for the one-off irfft
        if isinstance(result_vec_or_arr, list):
            return np.array(result_vec_or_arr, dtype=output_dtype)
        elif isinstance(result_vec_or_arr, np.ndarray):
            return result_vec_or_arr.astype(
                output_dtype, copy=False
            )  # Ensure correct real type
        else:
            raise TypeError(
                f"Unexpected return type from C++ irfft binding: {type(result_vec_or_arr)}"
            )
    except OmniDSPError as e:
        raise e
    except Exception as e:
        raise RuntimeError(f"IRFFT failed: {e}") from e


# --- Window Generation ---


def get_window(
    window_type: Union[WindowType, str],
    length: int,
    beta: Optional[float] = None,
    stddev: Optional[float] = None,
    dtype: Type[Union[np.float32, np.float64]] = np.float64,
    dsp_instance: Optional[OmniDSP] = None,
) -> np.ndarray:
    """
    Generates window coefficients.

    Args:
        window_type: The type of window (WindowType enum or string name).
        length: The desired window length. Must be non-negative.
        beta: The beta parameter (required for Kaiser window).
        stddev: The standard deviation parameter (required for Gaussian window).
        dtype: The desired data type (np.float32 or np.float64). Defaults to np.float64.
        dsp_instance: Optional pre-existing OmniDSP instance. If None, a temporary
                      Stub instance is used.

    Returns:
        np.ndarray: The window coefficients.

    Raises:
        OmniDSPError: If the C++ library encounters an error.
        TypeError: If input types are incompatible or unsupported.
        ValueError: If inputs are invalid (e.g., negative length, missing params).
        RuntimeError: If C++ bindings are not available or other unexpected errors occur.
    """
    if length < 0:
        raise ValueError("Window length cannot be negative.")
    if length == 0:
        return np.array([], dtype=dtype)  # Return empty array for length 0

    dsp = _get_dsp_instance(dsp_instance)

    # Convert string name to enum if necessary
    if isinstance(window_type, str):
        # Handle case variations gracefully
        wt_str_cap = window_type.capitalize()
        # Special case for Hann/Hanning? Assume Hann is standard.
        if wt_str_cap == "Hanning":
            wt_str_cap = "Hann"
        try:
            window_enum = WindowType[wt_str_cap]
        except KeyError:
            raise ValueError(f"Unknown window type string: '{window_type}'") from None
    elif isinstance(window_type, WindowType):
        window_enum = window_type
    else:
        raise TypeError("window_type must be a WindowType enum or string.")

    # Choose function based on type and dtype
    func_name = window_enum.name.lower() + "_window"  # e.g., "hann_window"

    try:
        cpp_func = getattr(dsp, func_name)
        args = [length]
        # Add parameters specific to certain windows
        if window_enum == WindowType.Kaiser:
            if beta is None:
                raise ValueError("beta parameter required for Kaiser window")
            args.append(beta)
        elif window_enum == WindowType.Gaussian:
            if stddev is None:
                raise ValueError("stddev parameter required for Gaussian window")
            args.append(stddev)

        # Call the correct C++ overload based on dtype
        if dtype == np.float32:
            # Cast params to float if needed
            args_casted = [int(args[0])] + [np.float32(p) for p in args[1:]]
            result_vec_or_arr = cpp_func(
                *args_casted
            )  # Assumes binding handles overload
        elif dtype == np.float64:
            args_casted = [int(args[0])] + [np.float64(p) for p in args[1:]]
            result_vec_or_arr = cpp_func(
                *args_casted
            )  # Assumes binding handles overload
        else:
            raise TypeError(f"Unsupported dtype for get_window: {dtype}")

        # Convert result if necessary
        if isinstance(result_vec_or_arr, list):
            return np.array(result_vec_or_arr, dtype=dtype)
        elif isinstance(result_vec_or_arr, np.ndarray):
            return result_vec_or_arr.astype(dtype, copy=False)
        else:
            raise TypeError(
                f"Unexpected return type from C++ {func_name} binding: {type(result_vec_or_arr)}"
            )

    except AttributeError:
        raise NotImplementedError(
            f"Window function '{func_name}' not bound or implemented in C++ backend."
        ) from None
    except OmniDSPError as e:
        raise e
    except Exception as e:
        raise RuntimeError(
            f"Window generation failed for type '{window_enum.name}': {e}"
        ) from e


# --- Plan Creation ---
# These functions REQUIRE an OmniDSP instance


def create_fft_plan(
    dsp_instance: OmniDSP,
    length: int,
    dtype: Type[Union[np.complex64, np.complex128]] = np.complex128,
) -> Plan:
    """Creates an optimized FFT plan."""
    _ensure_instance(dsp_instance)
    # Let C++ binding handle template instantiation based on dtype passed implicitly or explicitly
    # The return type hint Union is broad; actual type depends on dtype.
    return dsp_instance.create_fft_plan(
        length
    )  # Assume binding uses dtype info if overloaded


def create_rfft_plan(
    dsp_instance: OmniDSP,
    length: int,
    dtype: Type[Union[np.float32, np.float64]] = np.float64,
) -> Plan:
    """Creates an optimized real FFT plan."""
    _ensure_instance(dsp_instance)
    return dsp_instance.create_rfft_plan(length)  # Assume binding uses dtype info


def create_cqt_plan(
    dsp_instance: OmniDSP,
    sample_rate: float,
    min_freq: float,
    max_freq: float,
    bins_per_octave: int,
    dtype: Type[Union[np.float32, np.float64]] = np.float64,
) -> Plan:
    """Creates an optimized CQT plan."""
    _ensure_instance(dsp_instance)
    # Cast args if dtype is float32? The C++ template should handle this.
    # The binding needs to correctly call the float or double C++ template.
    # Let's assume the binding handles the dispatch based on a potential dtype argument
    # or by providing separate bindings like create_cqt_plan_float/double.
    # For now, assume the binding takes float/double directly.
    if dtype == np.float32:
        return dsp_instance.create_cqt_plan(
            np.float32(sample_rate),
            np.float32(min_freq),
            np.float32(max_freq),
            bins_per_octave,
        )
    else:
        return dsp_instance.create_cqt_plan(
            np.float64(sample_rate),
            np.float64(min_freq),
            np.float64(max_freq),
            bins_per_octave,
        )


def create_resample_plan(
    dsp_instance: OmniDSP,
    input_rate: float,
    output_rate: float,
    max_input_size: int = 0,
    dtype: Type[Union[np.float32, np.float64]] = np.float64,
) -> Plan:
    """Creates an optimized resampling plan."""
    _ensure_instance(dsp_instance)
    # Assume binding handles dtype dispatch
    return dsp_instance.create_resample_plan(input_rate, output_rate, max_input_size)


def create_convolution_plan(
    dsp_instance: OmniDSP,
    kernel_arr: ArrayLike,
    mode: ConvolutionMode = ConvolutionMode.Full,
) -> Plan:
    """Creates an optimized convolution plan."""
    _ensure_instance(dsp_instance)
    kernel_np = _ensure_numpy(kernel_arr)
    # Assume binding handles dtype dispatch
    return dsp_instance.create_convolution_plan(kernel_np, mode)


def create_correlation_plan(
    dsp_instance: OmniDSP,
    template_arr: ArrayLike,
    mode: ConvolutionMode = ConvolutionMode.Full,
) -> Plan:
    """Creates an optimized correlation plan."""
    _ensure_instance(dsp_instance)
    template_np = _ensure_numpy(template_arr)
    # Assume binding handles dtype dispatch
    return dsp_instance.create_correlation_plan(template_np, mode)


# --- Plan Execution ---
# Provide functions that take a plan and numpy arrays


def execute_plan(plan: Plan, input_arr: np.ndarray, output_arr: np.ndarray) -> None:
    """
    Executes a pre-computed Plan on input/output NumPy arrays.

    Args:
        plan: The Plan object created by a create_*_plan function.
        input_arr: NumPy array containing the input data. Must match plan's expected dtype.
        output_arr: NumPy array to store the output data. Must match plan's expected
                    output dtype and have sufficient size.

    Raises:
        OmniDSPError: If the C++ library encounters an error during execution.
        TypeError: If plan is not a valid Plan object or array dtypes are incorrect.
        ValueError: If array dimensions or sizes are incorrect.
        RuntimeError: If C++ bindings are not available or other unexpected errors occur.
    """
    if not _bindings_available:
        raise RuntimeError("OmniDSP C++ bindings are not available.")
    if not hasattr(plan, "execute"):  # Basic check for Plan-like object
        raise TypeError("Provided object is not a valid OmniDSP Plan.")

    # Basic input validation (more specific checks might be needed depending on plan type)
    if not isinstance(input_arr, np.ndarray):
        raise TypeError("input_arr must be a NumPy array.")
    if not isinstance(output_arr, np.ndarray):
        raise TypeError("output_arr must be a NumPy array.")
    if not output_arr.flags["WRITEABLE"]:
        raise ValueError("output_arr must be writeable.")
    if not input_arr.flags["C_CONTIGUOUS"]:
        warnings.warn(
            "Input array is not C-contiguous. A copy will be made.", RuntimeWarning
        )
        input_arr = np.ascontiguousarray(input_arr)
    if not output_arr.flags["C_CONTIGUOUS"]:
        warnings.warn(
            "Output array is not C-contiguous. Performance may be affected.",
            RuntimeWarning,
        )
        # Cannot force output to be contiguous if it wasn't passed that way. C++ binding must handle.

    try:
        # Let the bound method handle span conversion and more detailed type/size checks
        plan.execute(input_arr, output_arr)
    except OmniDSPError as e:
        raise e
    except Exception as e:
        # Catch potential pybind11 errors during argument conversion etc.
        raise RuntimeError(f"Plan execution failed: {e}") from e


# --- Internal Helper ---
def _ensure_instance(dsp_instance: Optional[OmniDSP]):
    """Raises TypeError if dsp_instance is not a valid OmniDSP object."""
    if not _bindings_available:
        raise RuntimeError("OmniDSP C++ bindings are not available.")
    if dsp_instance is None or not isinstance(dsp_instance, OmniDSP):
        raise TypeError("A valid OmniDSP instance is required for creating Plans.")


# --- Optional: Add aliases for convenience ---
# Use def instead of lambda to satisfy linters (E731)
def hann(length: int, **kwargs) -> np.ndarray:
    """Alias for get_window(WindowType.Hann, length, **kwargs)."""
    return get_window(WindowType.Hann, length, **kwargs)


def hamming(length: int, **kwargs) -> np.ndarray:
    """Alias for get_window(WindowType.Hamming, length, **kwargs)."""
    return get_window(WindowType.Hamming, length, **kwargs)


# Add other aliases as needed...
def kaiser(length: int, beta: float, **kwargs) -> np.ndarray:
    """Alias for get_window(WindowType.Kaiser, length, beta=beta, **kwargs)."""
    return get_window(WindowType.Kaiser, length, beta=beta, **kwargs)

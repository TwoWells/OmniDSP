# -*- coding: utf-8 -*-
"""
Type stub file for the OmniDSP Python API (`omnidsp_py.api`).

This file provides type hints for static analysis tools and IDEs.
It reflects the public interface defined in `api.py`.
"""

import numpy as np
from typing import Optional, Union, Sequence, Type, Any, overload, TypeVar

# --- Type Aliases ---
# Define ArrayLike to accept NumPy arrays or sequences of numbers
_Numeric = Union[float, complex]
_RealNum = float
_ComplexNum = complex
_Float = float  # Alias for Python's float (maps to double in C++)
_Float32 = np.float32
_Float64 = np.float64
_Complex64 = np.complex64
_Complex128 = np.complex128

# Generic NumPy array type hints
_T_real = TypeVar("_T_real", bound=np.floating[Any])
_T_complex = TypeVar("_T_complex", bound=np.complexfloating[Any])
_T_numeric = TypeVar("_T_numeric", bound=np.number[Any])

ArrayLike = Union[np.ndarray[Any, np.dtype[_Numeric]], Sequence[_Numeric]]
RealArrayLike = Union[np.ndarray[Any, np.dtype[_RealNum]], Sequence[_RealNum]]
ComplexArrayLike = Union[np.ndarray[Any, np.dtype[_ComplexNum]], Sequence[_ComplexNum]]

# --- Forward Declarations / Types from Bindings ---
# These represent the classes exposed by the _omnidsp_cpp module.

# Enums
class Backend:
    Stub: Backend
    Accelerate: Backend
    OneMKL: Backend
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> Any: ...

class Status:
    Success: Status
    Failure: Status
    InvalidArgument: Status
    InvalidOperation: Status
    AllocationError: Status
    BackendError: Status
    UnsupportedFeature: Status
    SizeMismatch: Status
    OutOfBounds: Status
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> Any: ...

class DataType:
    Real: DataType
    Complex: DataType
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> Any: ...

class Domain:
    Time: Domain
    Frequency: Domain
    Quefrency: Domain
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> Any: ...

class WindowType:
    Bartlett: WindowType
    Blackman: WindowType
    Flattop: WindowType
    Gaussian: WindowType
    Hamming: WindowType
    Hann: WindowType
    Kaiser: WindowType
    Rectangular: WindowType
    Triangular: WindowType
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> Any: ...

class ConvolutionMode:
    Full: ConvolutionMode
    Same: ConvolutionMode
    Valid: ConvolutionMode
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> Any: ...

# Custom Exception
class OmniDSPError(RuntimeError):
    status: Status
    def __init__(self, s: Status, message: str) -> None: ...
    def get_status(self) -> Status: ...

# WindowSpec (as bound from C++)
class WindowSpecFloat:
    def __init__(
        self, type: WindowType = ..., param: Optional[float] = ...
    ) -> None: ...
    def get_type(self) -> WindowType: ...
    def get_beta(self) -> Optional[float]: ...
    def get_stddev(self) -> Optional[float]: ...

class WindowSpecDouble:
    def __init__(
        self, type: WindowType = ..., param: Optional[float] = ...
    ) -> None: ...  # Use float for double
    def get_type(self) -> WindowType: ...
    def get_beta(self) -> Optional[float]: ...
    def get_stddev(self) -> Optional[float]: ...

# Plan Base Class (Conceptual)
class _BasePlan: ...  # Cannot define execute here easily due to type variations

# Concrete Plan Types (as bound from C++) - Define key methods
class FFTPlanFloatComplex(_BasePlan):
    def fft(
        self,
        input_arr: np.ndarray[Any, np.dtype[_Complex64]],
        output_arr: np.ndarray[Any, np.dtype[_Complex64]],
    ) -> None: ...
    def ifft(
        self,
        input_arr: np.ndarray[Any, np.dtype[_Complex64]],
        output_arr: np.ndarray[Any, np.dtype[_Complex64]],
    ) -> None: ...
    def get_length(self) -> int: ...

class FFTPlanDoubleComplex(_BasePlan):
    def fft(
        self,
        input_arr: np.ndarray[Any, np.dtype[_Complex128]],
        output_arr: np.ndarray[Any, np.dtype[_Complex128]],
    ) -> None: ...
    def ifft(
        self,
        input_arr: np.ndarray[Any, np.dtype[_Complex128]],
        output_arr: np.ndarray[Any, np.dtype[_Complex128]],
    ) -> None: ...
    def get_length(self) -> int: ...

class RFFTPlanFloat(_BasePlan):
    def rfft(
        self,
        input_arr: np.ndarray[Any, np.dtype[_Float32]],
        output_arr: np.ndarray[Any, np.dtype[_Complex64]],
    ) -> None: ...
    def irfft(
        self,
        input_arr: np.ndarray[Any, np.dtype[_Complex64]],
        output_arr: np.ndarray[Any, np.dtype[_Float32]],
    ) -> None: ...
    def get_length(self) -> int: ...

class RFFTPlanDouble(_BasePlan):
    def rfft(
        self,
        input_arr: np.ndarray[Any, np.dtype[_Float64]],
        output_arr: np.ndarray[Any, np.dtype[_Complex128]],
    ) -> None: ...
    def irfft(
        self,
        input_arr: np.ndarray[Any, np.dtype[_Complex128]],
        output_arr: np.ndarray[Any, np.dtype[_Float64]],
    ) -> None: ...
    def get_length(self) -> int: ...

class CQTPlanFloat(_BasePlan):
    def execute(
        self,
        input_arr: np.ndarray[Any, np.dtype[_Float32]],
        output_arr: np.ndarray[Any, np.dtype[_Complex64]],
    ) -> None: ...
    def get_num_bins(self) -> int: ...
    def get_num_output_frames(self, input_length: int) -> int: ...
    def get_hop_length(self) -> int: ...

class CQTPlanDouble(_BasePlan):
    def execute(
        self,
        input_arr: np.ndarray[Any, np.dtype[_Float64]],
        output_arr: np.ndarray[Any, np.dtype[_Complex128]],
    ) -> None: ...
    def get_num_bins(self) -> int: ...
    def get_num_output_frames(self, input_length: int) -> int: ...
    def get_hop_length(self) -> int: ...

class ConvolutionPlanFloat(_BasePlan):
    def execute(
        self,
        input_arr: np.ndarray[Any, np.dtype[_Float32]],
        output_arr: np.ndarray[Any, np.dtype[_Float32]],
    ) -> None: ...
    def get_kernel_length(self) -> int: ...
    def get_mode(self) -> ConvolutionMode: ...
    def get_output_length(self, input_length: int) -> int: ...

# Add ConvolutionPlanDouble, ConvolutionPlanFloatComplex, ConvolutionPlanDoubleComplex ...

class CorrelationPlanFloat(_BasePlan):
    def execute(
        self,
        input_arr: np.ndarray[Any, np.dtype[_Float32]],
        output_arr: np.ndarray[Any, np.dtype[_Float32]],
    ) -> None: ...
    def get_template_length(self) -> int: ...
    def get_mode(self) -> ConvolutionMode: ...
    def get_output_length(self, input_length: int) -> int: ...

# Add CorrelationPlanDouble, CorrelationPlanFloatComplex, CorrelationPlanDoubleComplex ...

class ResamplePlanFloat(_BasePlan):
    def execute(
        self,
        input_arr: np.ndarray[Any, np.dtype[_Float32]],
        output_arr: np.ndarray[Any, np.dtype[_Float32]],
    ) -> None: ...
    def get_input_rate(self) -> float: ...
    def get_output_rate(self) -> float: ...
    def get_output_length(self, input_length: int) -> int: ...

class ResamplePlanDouble(_BasePlan):
    def execute(
        self,
        input_arr: np.ndarray[Any, np.dtype[_Float64]],
        output_arr: np.ndarray[Any, np.dtype[_Float64]],
    ) -> None: ...
    def get_input_rate(self) -> float: ...
    def get_output_rate(self) -> float: ...
    def get_output_length(self, input_length: int) -> int: ...

# Union type for any Plan
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

# OmniDSP Class (as bound from C++)
class OmniDSP:
    # Static factory method
    @staticmethod
    def create(backend: Backend = ...) -> OmniDSP: ...

    # Instance methods
    def get_backend(self) -> Backend: ...

    # Use @overload for methods with different type signatures
    @overload
    def convolve(
        self,
        input_arr: np.ndarray[Any, np.dtype[_Float32]],
        kernel_arr: np.ndarray[Any, np.dtype[_Float32]],
        mode: ConvolutionMode,
    ) -> np.ndarray[Any, np.dtype[_Float32]]: ...
    @overload
    def convolve(
        self,
        input_arr: np.ndarray[Any, np.dtype[_Float64]],
        kernel_arr: np.ndarray[Any, np.dtype[_Float64]],
        mode: ConvolutionMode,
    ) -> np.ndarray[Any, np.dtype[_Float64]]: ...
    @overload
    def convolve(
        self,
        input_arr: np.ndarray[Any, np.dtype[_Complex64]],
        kernel_arr: np.ndarray[Any, np.dtype[_Complex64]],
        mode: ConvolutionMode,
    ) -> np.ndarray[Any, np.dtype[_Complex64]]: ...
    @overload
    def convolve(
        self,
        input_arr: np.ndarray[Any, np.dtype[_Complex128]],
        kernel_arr: np.ndarray[Any, np.dtype[_Complex128]],
        mode: ConvolutionMode,
    ) -> np.ndarray[Any, np.dtype[_Complex128]]: ...
    # Catch-all if needed, though overloads are preferred
    # def convolve(self, input_arr: np.ndarray, kernel_arr: np.ndarray, mode: ConvolutionMode) -> np.ndarray: ...
    @overload
    def correlate(
        self,
        input_arr: np.ndarray[Any, np.dtype[_Float32]],
        template_arr: np.ndarray[Any, np.dtype[_Float32]],
        mode: ConvolutionMode,
    ) -> np.ndarray[Any, np.dtype[_Float32]]: ...
    @overload
    def correlate(
        self,
        input_arr: np.ndarray[Any, np.dtype[_Float64]],
        template_arr: np.ndarray[Any, np.dtype[_Float64]],
        mode: ConvolutionMode,
    ) -> np.ndarray[Any, np.dtype[_Float64]]: ...
    @overload
    def correlate(
        self,
        input_arr: np.ndarray[Any, np.dtype[_Complex64]],
        template_arr: np.ndarray[Any, np.dtype[_Complex64]],
        mode: ConvolutionMode,
    ) -> np.ndarray[Any, np.dtype[_Complex64]]: ...
    @overload
    def correlate(
        self,
        input_arr: np.ndarray[Any, np.dtype[_Complex128]],
        template_arr: np.ndarray[Any, np.dtype[_Complex128]],
        mode: ConvolutionMode,
    ) -> np.ndarray[Any, np.dtype[_Complex128]]: ...
    @overload
    def fft(
        self, input_arr: np.ndarray[Any, np.dtype[_Complex64]]
    ) -> np.ndarray[Any, np.dtype[_Complex64]]: ...
    @overload
    def fft(
        self, input_arr: np.ndarray[Any, np.dtype[_Complex128]]
    ) -> np.ndarray[Any, np.dtype[_Complex128]]: ...
    @overload
    def ifft(
        self, input_arr: np.ndarray[Any, np.dtype[_Complex64]]
    ) -> np.ndarray[Any, np.dtype[_Complex64]]: ...
    @overload
    def ifft(
        self, input_arr: np.ndarray[Any, np.dtype[_Complex128]]
    ) -> np.ndarray[Any, np.dtype[_Complex128]]: ...
    @overload
    def rfft(
        self, input_arr: np.ndarray[Any, np.dtype[_Float32]]
    ) -> np.ndarray[Any, np.dtype[_Complex64]]: ...
    @overload
    def rfft(
        self, input_arr: np.ndarray[Any, np.dtype[_Float64]]
    ) -> np.ndarray[Any, np.dtype[_Complex128]]: ...
    @overload
    def irfft(
        self, input_arr: np.ndarray[Any, np.dtype[_Complex64]], output_length: int
    ) -> np.ndarray[Any, np.dtype[_Float32]]: ...
    @overload
    def irfft(
        self, input_arr: np.ndarray[Any, np.dtype[_Complex128]], output_length: int
    ) -> np.ndarray[Any, np.dtype[_Float64]]: ...

    # Window generation methods
    # Overloading based on return type isn't directly possible in Python stubs easily.
    # Use Union for return type or bind specific versions (e.g., hann_window_f32, hann_window_f64)
    # Using Union here for simplicity, matching api.py's get_window helper approach.
    def bartlett_window(
        self, length: int
    ) -> np.ndarray[Any, np.dtype[Union[_Float32, _Float64]]]: ...
    def blackman_window(
        self, length: int
    ) -> np.ndarray[Any, np.dtype[Union[_Float32, _Float64]]]: ...
    def flattop_window(
        self, length: int
    ) -> np.ndarray[Any, np.dtype[Union[_Float32, _Float64]]]: ...
    def gaussian_window(
        self, length: int, stddev: float
    ) -> np.ndarray[Any, np.dtype[Union[_Float32, _Float64]]]: ...
    def hamming_window(
        self, length: int
    ) -> np.ndarray[Any, np.dtype[Union[_Float32, _Float64]]]: ...
    def hann_window(
        self, length: int
    ) -> np.ndarray[Any, np.dtype[Union[_Float32, _Float64]]]: ...
    def kaiser_window(
        self, length: int, beta: float
    ) -> np.ndarray[Any, np.dtype[Union[_Float32, _Float64]]]: ...
    def rectangular_window(
        self, length: int
    ) -> np.ndarray[Any, np.dtype[Union[_Float32, _Float64]]]: ...
    def triangular_window(
        self, length: int
    ) -> np.ndarray[Any, np.dtype[Union[_Float32, _Float64]]]: ...

    # Plan factory methods
    # Need to specify return types more accurately based on bound versions
    @overload
    def create_fft_plan(
        self, length: int
    ) -> (
        FFTPlanFloatComplex
    ): ...  # Assume binding resolves based on float/double preference or default
    @overload
    def create_fft_plan(self, length: int) -> FFTPlanDoubleComplex: ...
    @overload
    def create_rfft_plan(self, length: int) -> RFFTPlanFloat: ...
    @overload
    def create_rfft_plan(self, length: int) -> RFFTPlanDouble: ...
    @overload
    def create_cqt_plan(
        self, sample_rate: float, min_freq: float, max_freq: float, bins_per_octave: int
    ) -> CQTPlanFloat: ...
    @overload
    def create_cqt_plan(
        self, sample_rate: float, min_freq: float, max_freq: float, bins_per_octave: int
    ) -> CQTPlanDouble: ...
    @overload
    def create_resample_plan(
        self, input_rate: float, output_rate: float, max_input_size: int
    ) -> ResamplePlanFloat: ...
    @overload
    def create_resample_plan(
        self, input_rate: float, output_rate: float, max_input_size: int
    ) -> ResamplePlanDouble: ...

    # Convolution/Correlation plans need overloads for all 4 types (f32, f64, c64, c128)
    # Example for convolution:
    @overload
    def create_convolution_plan(
        self, kernel_arr: np.ndarray[Any, np.dtype[_Float32]], mode: ConvolutionMode
    ) -> ConvolutionPlanFloat: ...
    # Add other overloads for create_convolution_plan and create_correlation_plan

# --- High-Level API Functions (from api.py) ---

def create_dsp_instance(backend: Backend = ...) -> OmniDSP: ...
def convolve(
    input_arr: ArrayLike,
    kernel_arr: ArrayLike,
    mode: ConvolutionMode = ...,
    dsp_instance: Optional[OmniDSP] = ...,
) -> np.ndarray[Any, np.dtype[_Numeric]]: ...
def correlate(
    input_arr: ArrayLike,
    template_arr: ArrayLike,
    mode: ConvolutionMode = ...,
    dsp_instance: Optional[OmniDSP] = ...,
) -> np.ndarray[Any, np.dtype[_Numeric]]: ...
def fft(
    input_arr: ArrayLike, dsp_instance: Optional[OmniDSP] = ...
) -> np.ndarray[Any, np.dtype[_Complex128]]: ...
def ifft(
    input_arr: ComplexArrayLike, dsp_instance: Optional[OmniDSP] = ...
) -> np.ndarray[Any, np.dtype[_ComplexNum]]: ...
def rfft(
    input_arr: RealArrayLike, dsp_instance: Optional[OmniDSP] = ...
) -> np.ndarray[Any, np.dtype[_ComplexNum]]: ...
def irfft(
    input_arr: ComplexArrayLike,
    n: Optional[int] = ...,
    dsp_instance: Optional[OmniDSP] = ...,
) -> np.ndarray[Any, np.dtype[_RealNum]]: ...
def get_window(
    window_type: Union[WindowType, str],
    length: int,
    beta: Optional[float] = ...,
    stddev: Optional[float] = ...,
    dtype: Type[Union[np.float32, np.float64]] = ...,
    dsp_instance: Optional[OmniDSP] = ...,
) -> np.ndarray[Any, np.dtype[Union[np.float32, np.float64]]]: ...

# Plan creation functions require an OmniDSP instance
def create_fft_plan(
    dsp_instance: OmniDSP,
    length: int,
    dtype: Type[Union[np.complex64, np.complex128]] = ...,
) -> Plan: ...
def create_rfft_plan(
    dsp_instance: OmniDSP, length: int, dtype: Type[Union[np.float32, np.float64]] = ...
) -> Plan: ...
def create_cqt_plan(
    dsp_instance: OmniDSP,
    sample_rate: float,
    min_freq: float,
    max_freq: float,
    bins_per_octave: int,
    dtype: Type[Union[np.float32, np.float64]] = ...,
) -> Plan: ...
def create_resample_plan(
    dsp_instance: OmniDSP,
    input_rate: float,
    output_rate: float,
    max_input_size: int = ...,
    dtype: Type[Union[np.float32, np.float64]] = ...,
) -> Plan: ...
def create_convolution_plan(
    dsp_instance: OmniDSP, kernel_arr: ArrayLike, mode: ConvolutionMode = ...
) -> Plan: ...
def create_correlation_plan(
    dsp_instance: OmniDSP, template_arr: ArrayLike, mode: ConvolutionMode = ...
) -> Plan: ...

# Plan execution
def execute_plan(
    plan: Plan, input_arr: np.ndarray[Any, Any], output_arr: np.ndarray[Any, Any]
) -> None: ...

# Convenience aliases
def hann(
    length: int, **kwargs: Any
) -> np.ndarray[Any, np.dtype[Union[np.float32, np.float64]]]: ...
def hamming(
    length: int, **kwargs: Any
) -> np.ndarray[Any, np.dtype[Union[np.float32, np.float64]]]: ...
def kaiser(
    length: int, beta: float, **kwargs: Any
) -> np.ndarray[Any, np.dtype[Union[np.float32, np.float64]]]: ...

# ... etc ...

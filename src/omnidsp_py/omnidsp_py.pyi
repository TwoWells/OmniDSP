# omnidsp_py/omnidsp_py.pyi
# This file provides type hints for the omnidsp_py C++ extension module.
# It helps with static analysis and code completion in Python IDEs.

from typing import TypeVar, Union, overload, Callable, Any
import numpy as np
from enum import Enum

# --- Type Aliases ---
# Define type hints for common NumPy array types used in the library.
# Note: Using generic np.ndarray as precise dtype hinting within generics can be complex.
# The function/method docstrings and runtime checks provide specific dtype info.
_Float = TypeVar("_Float", bound=np.floating)
_Complex = TypeVar("_Complex", bound=np.complexfloating)
ArrayF32 = np.ndarray[Any, np.dtype[np.float32]]
ArrayF64 = np.ndarray[Any, np.dtype[np.float64]]
ArrayC64 = np.ndarray[Any, np.dtype[np.complex64]]
ArrayC128 = np.ndarray[Any, np.dtype[np.complex128]]
ArrayReal = Union[ArrayF32, ArrayF64]
ArrayComplex = Union[ArrayC64, ArrayC128]
# Type variable for the window function callable passed to CQTPlan
T_WindowFunc = Callable[[np.ndarray], np.ndarray]

# --- Enums ---
# Define Python Enum classes corresponding to the C++ enums.

class Direction(Enum):
    """Specifies the direction of the Fourier Transform."""

    FORWARD: Direction = ...
    INVERSE: Direction = ...

class Precision(Enum):
    """Specifies the floating-point precision for calculations."""

    SINGLE: Precision = ...
    DOUBLE: Precision = ...

class Domain(Enum):
    """Specifies the domain of the input/output signals."""

    COMPLEX: Domain = ...
    REAL: Domain = ...

class NormMode(Enum):
    """Specifies the normalization/scaling mode applied to the transforms."""

    BACKWARD: NormMode = ...
    ORTHO: NormMode = ...
    FORWARD: NormMode = ...

# --- Plan Classes ---
# Define the classes representing the precomputed plans.

class FFTPlanFloat:
    """Manages a pre-calculated plan for efficient FFT execution (float precision)."""

    def __init__(
        self,
        length: int,
        precision: Precision,
        direction: Direction,
        domain: Domain,
        norm: NormMode = NormMode.BACKWARD,
    ) -> None: ...
    def getLength(self) -> int: ...
    def getComplexLength(self) -> int: ...
    def getDirection(self) -> Direction: ...
    def getPrecision(self) -> Precision: ...
    def getDomain(self) -> Domain: ...
    def getNormMode(self) -> NormMode: ...
    # Note: execute methods are not typically called directly on the plan from Python;
    # use the convenience functions (fft, ifft, etc.) instead.

class FFTPlanDouble:
    """Manages a pre-calculated plan for efficient FFT execution (double precision)."""

    def __init__(
        self,
        length: int,
        precision: Precision,
        direction: Direction,
        domain: Domain,
        norm: NormMode = NormMode.BACKWARD,
    ) -> None: ...
    def getLength(self) -> int: ...
    def getComplexLength(self) -> int: ...
    def getDirection(self) -> Direction: ...
    def getPrecision(self) -> Precision: ...
    def getDomain(self) -> Domain: ...
    def getNormMode(self) -> NormMode: ...
    # Note: execute methods are not typically called directly on the plan from Python.

class CQTPlanFloat:
    """Manages a pre-calculated plan for efficient recursive CQT execution (float precision)."""

    def __init__(
        self,
        sample_rate: float,
        hop_length: int,
        lowest_freq: float,
        highest_freq: float,
        bins_per_octave: int,
        window_function: T_WindowFunc,
        sparsity_threshold: float = 1e-5,
        fir_filter_order: int = 101,
    ) -> None: ...
    def execute(
        self, input: ArrayF32
    ) -> ArrayC64: ...  # Takes 1D real, returns 2D complex
    def getNumBins(self) -> int: ...
    def getSampleRate(self) -> float: ...
    def getHopLength(self) -> int: ...
    def getLowestFrequency(self) -> float: ...
    def getHighestFrequency(self) -> float: ...
    def getBinsPerOctave(self) -> int: ...
    def getNumOctaves(self) -> int: ...
    def getSparsityThreshold(self) -> float: ...
    def getFirFilterOrder(self) -> int: ...
    # def getFFTLength(self) -> int: ... # Old getter, might be replaced by getOctaveFFTLengths if bound
    # def getOctaveFFTLengths(self) -> list[int]: ... # Example if bound

class CQTPlanDouble:
    """Manages a pre-calculated plan for efficient recursive CQT execution (double precision)."""

    def __init__(
        self,
        sample_rate: float,
        hop_length: int,
        lowest_freq: float,
        highest_freq: float,
        bins_per_octave: int,
        window_function: T_WindowFunc,
        sparsity_threshold: float = 1e-5,
        fir_filter_order: int = 101,
    ) -> None: ...
    def execute(
        self, input: ArrayF64
    ) -> ArrayC128: ...  # Takes 1D real, returns 2D complex
    def getNumBins(self) -> int: ...
    def getSampleRate(self) -> float: ...
    def getHopLength(self) -> int: ...
    def getLowestFrequency(self) -> float: ...
    def getHighestFrequency(self) -> float: ...
    def getBinsPerOctave(self) -> int: ...
    def getNumOctaves(self) -> int: ...
    def getSparsityThreshold(self) -> float: ...
    def getFirFilterOrder(self) -> int: ...
    # def getFFTLength(self) -> int: ... # Old getter
    # def getOctaveFFTLengths(self) -> list[int]: ... # Example if bound

# --- Window Class ---
# Define the Window class with its overloaded static methods.

class Window:
    """Provides common window functions for signal processing."""
    @staticmethod
    @overload
    def hann(input: ArrayF32) -> ArrayF32: ...
    @staticmethod
    @overload
    def hann(input: ArrayF64) -> ArrayF64: ...
    @staticmethod
    def hann(input: ArrayReal) -> ArrayReal: ...  # Implementation signature
    @staticmethod
    @overload
    def hamming(input: ArrayF32) -> ArrayF32: ...
    @staticmethod
    @overload
    def hamming(input: ArrayF64) -> ArrayF64: ...
    @staticmethod
    def hamming(input: ArrayReal) -> ArrayReal: ...  # Implementation signature
    @staticmethod
    @overload
    def kaiser(input: ArrayF32, beta: float) -> ArrayF32: ...
    @staticmethod
    @overload
    def kaiser(input: ArrayF64, beta: float) -> ArrayF64: ...
    @staticmethod
    def kaiser(
        input: ArrayReal, beta: float
    ) -> ArrayReal: ...  # Implementation signature
    @staticmethod
    @overload
    def flattop(input: ArrayF32) -> ArrayF32: ...
    @staticmethod
    @overload
    def flattop(input: ArrayF64) -> ArrayF64: ...
    @staticmethod
    def flattop(input: ArrayReal) -> ArrayReal: ...  # Implementation signature

# --- Convenience Functions ---
# Define the module-level convenience functions using @overload for type hinting.

@overload
def fft(input: ArrayC64) -> ArrayC64: ...
@overload
def fft(input: ArrayC128) -> ArrayC128: ...
def fft(input: ArrayComplex) -> ArrayComplex: ...  # Implementation signature
@overload
def ifft(input: ArrayC64) -> ArrayC64: ...
@overload
def ifft(input: ArrayC128) -> ArrayC128: ...
def ifft(input: ArrayComplex) -> ArrayComplex: ...  # Implementation signature
@overload
def rfft(real_input: ArrayF32) -> ArrayC64: ...
@overload
def rfft(real_input: ArrayF64) -> ArrayC128: ...
def rfft(real_input: ArrayReal) -> ArrayComplex: ...  # Implementation signature
@overload
def irfft(complex_input: ArrayC64) -> ArrayF32: ...
@overload
def irfft(complex_input: ArrayC128) -> ArrayF64: ...
def irfft(complex_input: ArrayComplex) -> ArrayReal: ...  # Implementation signature

# --- Version Attribute ---
__version__: str = ...

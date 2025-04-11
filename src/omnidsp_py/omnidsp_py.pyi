# -*- coding: utf-8 -*-
# Stubs for the omnidsp_py C++ extension module
# Generated based on OmniDSP/python/bindings.cpp

from enum import Enum
from typing import Callable, TypeAlias
import numpy as np
from numpy.typing import NDArray

# --- Type Aliases for NumPy Arrays ---
Float32Array: TypeAlias = NDArray[np.float32]
Float64Array: TypeAlias = NDArray[np.float64]
Complex64Array: TypeAlias = NDArray[np.complex64]
Complex128Array: TypeAlias = NDArray[np.complex128]

# --- Module Attributes ---
__version__: str
__doc__: str | None

# --- Enums ---
class Direction(Enum):
    FORWARD: Direction = ...
    INVERSE: Direction = ...
    def __int__(self) -> int: ...

class Precision(Enum):
    SINGLE: Precision = ...
    DOUBLE: Precision = ...
    def __int__(self) -> int: ...

class Domain(Enum):
    COMPLEX: Domain = ...
    REAL: Domain = ...
    def __int__(self) -> int: ...

class NormMode(Enum):
    BACKWARD: NormMode = ...
    ORTHO: NormMode = ...
    FORWARD: NormMode = ...
    def __int__(self) -> int: ...

# --- Classes ---
class FFTPlanFloat:
    """
    Manages a pre-calculated plan for efficient FFT execution (float precision).
    """
    def __init__(
        self,
        length: int,
        precision: Precision,
        direction: Direction,
        domain: Domain,
        norm: NormMode = NormMode.BACKWARD,
    ) -> None:
        """
        Constructs and initializes an FFT plan.

        Args:
            length (int): The size 'N' of the transform.
                - For Domain.COMPLEX: Number of complex points.
                - For Domain.REAL: Number of real points (complex size will be N/2 + 1).
                - Note: Accelerate backend requires N to be power-of-2 for REAL domain.
            precision (Precision): Must be Precision.SINGLE for FFTPlanFloat.
            direction (Direction): Transform direction (Direction.FORWARD or Direction.INVERSE).
            domain (Domain): Transform domain (Domain.COMPLEX or Domain.REAL).
            norm (NormMode, optional): Normalization mode. Defaults to NormMode.BACKWARD.
        """
        ...
    def getLength(self) -> int:
        """Gets the length 'N' associated with the plan."""
        ...
    def getComplexLength(self) -> int:
        """
        Gets the length (Nc = N/2 + 1) of the complex spectrum for Domain.REAL plans.
        Returns N for Domain.COMPLEX plans.
        """
        ...
    def getDirection(self) -> Direction:
        """Gets the transform direction configured for this plan."""
        ...
    def getPrecision(self) -> Precision:
        """Gets the floating-point precision configured for this plan."""
        ...
    def getDomain(self) -> Domain:
        """Gets the transform domain configured for this plan."""
        ...
    def getNormMode(self) -> NormMode:
        """Gets the normalization mode configured for this plan."""
        ...

class FFTPlanDouble:
    """
    Manages a pre-calculated plan for efficient FFT execution (double precision).
    """
    def __init__(
        self,
        length: int,
        precision: Precision,
        direction: Direction,
        domain: Domain,
        norm: NormMode = NormMode.BACKWARD,
    ) -> None:
        """
        Constructs and initializes an FFT plan.

        Args:
            length (int): The size 'N' of the transform.
            precision (Precision): Must be Precision.DOUBLE for FFTPlanDouble.
            direction (Direction): Transform direction.
            domain (Domain): Transform domain.
            norm (NormMode, optional): Normalization mode. Defaults to NormMode.BACKWARD.
        """
        ...
    def getLength(self) -> int:
        """Gets the length 'N' associated with the plan."""
        ...
    def getComplexLength(self) -> int:
        """
        Gets the length (Nc = N/2 + 1) of the complex spectrum for Domain.REAL plans.
        Returns N for Domain.COMPLEX plans.
        """
        ...
    def getDirection(self) -> Direction:
        """Gets the transform direction configured for this plan."""
        ...
    def getPrecision(self) -> Precision:
        """Gets the floating-point precision configured for this plan."""
        ...
    def getDomain(self) -> Domain:
        """Gets the transform domain configured for this plan."""
        ...
    def getNormMode(self) -> NormMode:
        """Gets the normalization mode configured for this plan."""
        ...

# Type hint for the window function callable passed to CQTPlan
# It takes an array (often just size 1 from C++) and should return window coefficients
WindowFuncFloat: TypeAlias = Callable[[Float32Array], Float32Array]
WindowFuncDouble: TypeAlias = Callable[[Float64Array], Float64Array]

class CQTPlanFloat:
    """
    Manages a pre-calculated plan for efficient Constant Q Transform (CQT) execution (float precision).
    """
    def __init__(
        self,
        sample_rate: float,
        lowest_freq: float,
        highest_freq: float,
        bins_per_octave: int,
        window_function: WindowFuncFloat,
    ) -> None:
        """
        Constructs and initializes a CQT plan.

        Args:
            sample_rate (float): Sample rate of the input signal in Hz.
            lowest_freq (float): Lowest frequency of interest (Hz, must be > 0).
            highest_freq (float): Highest frequency of interest (Hz, must be <= sample_rate / 2).
            bins_per_octave (int): Number of frequency bins per octave (must be > 0).
            window_function (callable): A Python function that accepts a NumPy array (placeholder)
                and returns a NumPy array containing the window coefficients of the appropriate length
                as determined internally by the CQT algorithm for the current bin.
        """
        ...
    def execute(self, input: Float32Array) -> Complex64Array:
        """
        Executes the CQT transform.

        Args:
            input (numpy.ndarray[float32]): A 1D NumPy array containing the input signal.

        Returns:
            numpy.ndarray[complex64]: A 1D NumPy array containing the complex CQT coefficients.
        """
        ...
    def getNumBins(self) -> int:
        """Gets the number of CQT frequency bins."""
        ...
    def getSampleRate(self) -> float:
        """Gets the sample rate used for this plan (Hz)."""
        ...
    def getLowestFrequency(self) -> float:
        """Gets the lowest frequency configured for this plan (Hz)."""
        ...
    def getHighestFrequency(self) -> float:
        """Gets the highest frequency configured for this plan (Hz)."""
        ...
    def getBinsPerOctave(self) -> int:
        """Gets the number of bins per octave configured for this plan."""
        ...
    def getFFTLength(self) -> int:
        """Gets the length of the internal FFT used by the plan."""
        ...

class CQTPlanDouble:
    """
    Manages a pre-calculated plan for efficient CQT execution (double precision).
    """
    def __init__(
        self,
        sample_rate: float,
        lowest_freq: float,
        highest_freq: float,
        bins_per_octave: int,
        window_function: WindowFuncDouble,
    ) -> None:
        """
        Constructs and initializes a CQT plan.

        Args:
            sample_rate (float): Sample rate in Hz.
            lowest_freq (float): Lowest frequency in Hz.
            highest_freq (float): Highest frequency in Hz.
            bins_per_octave (int): Number of bins per octave.
            window_function (callable): Python function returning window coefficients.
        """
        ...
    def execute(self, input: Float64Array) -> Complex128Array:
        """
        Executes the CQT transform.

        Args:
            input (numpy.ndarray[float64]): A 1D NumPy array containing the input signal.

        Returns:
            numpy.ndarray[complex128]: A 1D NumPy array containing the complex CQT coefficients.
        """
        ...
    def getNumBins(self) -> int:
        """Gets the number of CQT frequency bins."""
        ...
    def getSampleRate(self) -> float:
        """Gets the sample rate used for this plan (Hz)."""
        ...
    def getLowestFrequency(self) -> float:
        """Gets the lowest frequency configured for this plan (Hz)."""
        ...
    def getHighestFrequency(self) -> float:
        """Gets the highest frequency configured for this plan (Hz)."""
        ...
    def getBinsPerOctave(self) -> int:
        """Gets the number of bins per octave configured for this plan."""
        ...
    def getFFTLength(self) -> int:
        """Gets the length of the internal FFT used by the plan."""
        ...

class Window:
    """Provides common window functions for signal processing."""
    @staticmethod
    def hann_float(input: Float32Array) -> Float32Array:
        """Applies the Hann window (float precision)."""
        ...
    @staticmethod
    def hann_double(input: Float64Array) -> Float64Array:
        """Applies the Hann window (double precision)."""
        ...
    @staticmethod
    def hamming_float(input: Float32Array) -> Float32Array:
        """Applies the Hamming window (float precision)."""
        ...
    @staticmethod
    def hamming_double(input: Float64Array) -> Float64Array:
        """Applies the Hamming window (double precision)."""
        ...
    @staticmethod
    def kaiser_float(input: Float32Array, beta: float) -> Float32Array:
        """Applies the Kaiser window (float precision)."""
        ...
    @staticmethod
    def kaiser_double(input: Float64Array, beta: float) -> Float64Array:
        """Applies the Kaiser window (double precision)."""
        ...
    @staticmethod
    def flattop_float(input: Float32Array) -> Float32Array:
        """Applies the Flat-top window (float precision)."""
        ...
    @staticmethod
    def flattop_double(input: Float64Array) -> Float64Array:
        """Applies the Flat-top window (double precision)."""
        ...

# --- Convenience Functions ---
def fft_float(input: Complex64Array) -> Complex64Array:
    """Performs C2C forward FFT (float, out-of-place, NormMode.BACKWARD)."""
    ...
def fft_double(input: Complex128Array) -> Complex128Array:
    """Performs C2C forward FFT (double, out-of-place, NormMode.BACKWARD)."""
    ...
def ifft_float(input: Complex64Array) -> Complex64Array:
    """Performs C2C inverse FFT (float, out-of-place, NormMode.BACKWARD)."""
    ...
def ifft_double(input: Complex128Array) -> Complex128Array:
    """Performs C2C inverse FFT (double, out-of-place, NormMode.BACKWARD)."""
    ...
def rfft_float(real_input: Float32Array) -> Complex64Array:
    """Performs R2C forward FFT (float, out-of-place, NormMode.BACKWARD). Returns N/2+1 complex points."""
    ...
def rfft_double(real_input: Float64Array) -> Complex128Array:
    """Performs R2C forward FFT (double, out-of-place, NormMode.BACKWARD). Returns N/2+1 complex points."""
    ...
def irfft_float(complex_input: Complex64Array) -> Float32Array:
    """Performs C2R inverse FFT (float, out-of-place, NormMode.BACKWARD). Input must have Hermitian symmetry."""
    ...
def irfft_double(complex_input: Complex128Array) -> Float64Array:
    """Performs C2R inverse FFT (double, out-of-place, NormMode.BACKWARD). Input must have Hermitian symmetry."""
    ...

# Note: In-place convenience functions (fft_inplace, ifft_inplace) were not bound in bindings.cpp
# __init__.py inside src/omnidsp_py

"""
OmniDSP Python Bindings Package.

Provides access to OmniDSP C++ functionality, including FFT and CQT plans,
convenience functions, and windowing utilities, accessible via Python wrappers
that automatically handle data type dispatching (float32/float64).
"""

# --- Import core C++ components needed directly or by wrappers ---
from .omnidsp_py import (
    # Enums
    Direction,
    Precision,
    Domain,
    NormMode,
    # Plan Classes (separate types for float/double)
    FFTPlanFloat,
    FFTPlanDouble,
    CQTPlanFloat,
    CQTPlanDouble,
    # Window Class (contains overloaded C++ static methods)
    Window as _Window_cpp,  # Import C++ bound Window with a temporary name
)

# --- Import Python wrappers and factories from api.py ---
# These provide the user-facing, overloaded API for functions/factories
from .api import fft, ifft, rfft, irfft, create_fft_plan, create_cqt_plan

# --- Expose the Window class directly ---
# Re-export the C++ bound class using the public 'Window' name.
# This pattern (import as _, then assign) is used for explicitness
# and to avoid potential future namespace collisions if other modules
# imported here also defined 'Window'.
Window = _Window_cpp

# --- Define Public API using __all__ ---
# Lists everything intended for public use.
__all__ = [
    # Enums
    "Direction",
    "Precision",
    "Domain",
    "NormMode",
    # Plan Classes (Exposing both for direct use and type hinting)
    "FFTPlanFloat",
    "FFTPlanDouble",
    "CQTPlanFloat",
    "CQTPlanDouble",
    # Window Class (providing access to .hann(), .hamming(), etc.)
    "Window",
    # Overloaded Convenience Functions (from api.py)
    "fft",
    "ifft",
    "rfft",
    "irfft",
    # Factory Functions (from api.py)
    "create_fft_plan",
    "create_cqt_plan",
]

# --- Version (optional) ---
try:
    # Try importing __version__ defined in the C++ module
    from .omnidsp_py import __version__  # noqa: F401

    __all__.append("__version__")
except ImportError:
    # If not defined, don't add __version__ to __all__
    pass

# --- Optional: Clean up temporary import ---
# Delete the temporary _Window_cpp name from the module's namespace.
try:
    del _Window_cpp
except NameError:
    pass

# __init__.py inside the 'python' source directory (will be installed to site-packages/omnidsp_py)

import os
import sys
import warnings

# Add the directory containing this __init__.py file (which also contains the bundled DLLs)
# to the DLL search path on Windows Python 3.8+
# if sys.platform == 'win32' and sys.version_info >= (3, 8):
#     _package_dir = os.path.dirname(__file__)
#     try:
#         # Check if the directory exists and add it
#         if os.path.exists(_package_dir):
#             os.add_dll_directory(_package_dir)
#             # Optional: print statement for debugging installation/import
#             print(f"DEBUG [omnidsp_py init]: Added DLL directory: {_package_dir}", file=sys.stderr)
#         else:
#              warnings.warn(f"OmniDSP package directory not found: {_package_dir}", ImportWarning)
# 
#     except FileNotFoundError:
#         # This might happen in unusual circumstances
#         warnings.warn(f"OmniDSP package directory not found during DLL path setup: {_package_dir}", ImportWarning)
#     except OSError as e:
#         # This might happen if the path is invalid or other OS error occurs
#          warnings.warn(f"OSError adding OmniDSP DLL directory {_package_dir}: {e}", ImportWarning)
#     except Exception as e:
#         # Catch any other unexpected errors during path addition
#          warnings.warn(f"Unexpected error adding OmniDSP DLL directory {_package_dir}: {e}", ImportWarning)


# --- Crucial Step: Import and expose symbols from the compiled C++ module ---
# The actual .pyd file might have a complex name like omnidsp_py.cp313-win_amd64.pyd
# but Python usually allows importing it using the base name defined in PYBIND11_MODULE.
# We import everything from the C++ module into the package's namespace.
try:
    # Assuming the PYBIND11_MODULE name in bindings.cpp is 'omnidsp_py'
    from .omnidsp_py import * # Optionally, clean up namespace if needed, though `import *` is common here
    # Example: clean up _package_dir if you don't want it accessible
    # del _package_dir 
    # del os
    # del sys
    # del warnings

except ImportError as e:
    # If the underlying .pyd import still fails even after adding the DLL dir,
    # raise a more informative error or re-raise the original one.
    if "DLL load failed" in str(e):
         raise ImportError(
             "DLL load failed while importing OmniDSP's C++ core module. "
             "Ensure bundled dependencies (MKL, OpenMP, TBB) are present and compatible, "
             "and that the correct MSVC++ Redistributable is installed."
         ) from e
    else:
         raise # Re-raise other import errors
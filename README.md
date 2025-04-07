# OmniDSP - High-Performance Cross-Platform DSP Library

## Overview

This project provides a high-performance C++ library for Digital Signal Processing (DSP). Its primary goal is to offer a consistent, easy-to-use API while leveraging highly optimized, platform-specific backends for maximum performance across various systems. It currently includes Fast Fourier Transform (FFT) and Constant-Q Transform (CQT) functionality, with plans to expand to a broader range of DSP functions. It automatically selects between Intel's oneAPI Math Kernel Library (oneMKL) or Apple's Accelerate framework where available to utilize hardware acceleration.

If neither backend is detected, a stub implementation is compiled which will throw runtime errors upon use, indicating that a functional backend was not available during the build.

## Features

*   **Cross-Platform Performance:** Designed for high-performance DSP across platforms, utilizing optimized backends.
*   **Platform Abstraction:** Single C++ API for DSP operations, simplifying development.
*   **Optimized Backends:** Uses Intel oneMKL (via oneAPI) or Apple Accelerate framework for hardware-accelerated FFTs.
*   **Automatic Backend Selection:** CMake build system detects and selects the appropriate backend during the build process.
*   **Python Bindings:** Provides Python bindings via pybind11 for easy integration with NumPy and the Python ecosystem.
*   **Precision Support:** Works with `float` (single) and `double` (double) precision values.
*   **Transform Types:** Supports Complex-to-Complex (C2C) and Real-to-Complex/Complex-to-Real (R2C/C2R) FFTs.
*   **Execution Modes:** Supports in-place and out-of-place execution for C2C transforms.
*   **Configurable Normalization:** Supports different scaling conventions for FFTs (`NormMode`).
*   **Plan-Based Interface:** `FFTPlan` class allows pre-computation for efficient execution of multiple FFTs.
*   **Convenience Functions:** Provides simple functions for common FFT operations using `std::vector`.
*   **Window Functions:** Includes implementations for common window functions like Hann, Hamming, Kaiser, and Flat-top.
*   **Build System:** Uses CMake for cross-platform building and dependency management.
*   **Installation Support:** Provides CMake installation rules and package configuration files for easy integration into other CMake projects.
*   **Unit Testing:** Includes a test suite using GoogleTest.

## Project Structure

```
OmniDSP/
│
├── CMakeLists.txt          # Main CMake build script
├── Config.cmake.in         # Template for CMake package config file
│
├── examples/               # Usage examples
│   ├── cpp/                # C++ examples
│   │   ├── CMakeLists.txt  # CMake script for C++ examples
│   │   ├── cqt.cpp
│   │   └── ...
│   └── python/             # Python examples
│       ├── requirements.txt
│       ├── run_fft_example.py
│       └── ...
│
├── include/                # Public header files
│   └── OmniDSP/
│       ├── omnidsp.h         # Main header (include )
│       ├── cqt.h             # CQT specific header
│       └── windows.h         # Windowing functions header
│
├── src/                    # Implementation source files
│   ├── omnidsp.cpp           # Platform-independent convenience functions
│   ├── cqt.cpp               # CQTPlan implementation using FFTPlan
│   ├── bindings.cpp          # Python bindings (pybind11) source
│   └── backend/                # Backend-specific implementations
│       ├── onemkl.cpp          # oneMKL specific implementation
│       ├── accelerate.cpp      # Apple Accelerate specific implementation
│       └── stub.cpp            # Fallback/error implementation
│
├── tests/                  # Unit tests
│   ├── CMakeLists.txt
│   └── *.cpp
│
├── build/                  # (Generated Directory) Build artifacts
│
└── README.md               # This file (or its source)
```

## Prerequisites

1.  **CMake:** Version 3.15 or later (3.21+ recommended).
2.  **C++ Compiler:** A compiler supporting C++17 (e.g., GCC, Clang, MSVC).
3.  **FFT Backend (at least one for full FFT functionality):**
    *   **Intel oneMKL:** Install Intel oneAPI Base Toolkit. Required if building/running on Windows/Linux, or macOS with MKL preference.
    *   **Apple Accelerate:** On macOS, included with Xcode and Command Line Tools. Used by default on macOS unless MKL is preferred.
4.  **Python:** Version 3.x (for Python bindings and examples).
5.  **Python Packages:** `numpy` is required at runtime to use the Python bindings and examples. Other packages like `matplotlib` might be used in specific examples. See `examples/python/requirements.txt`.
6.  **Git:** Required by CMake's `FetchContent` to download GoogleTest and pybind11.

## Development Environment Setup (VS Code)

For the best C++ development experience with IntelliSense (autocompletion, error checking, code navigation) in Visual Studio Code, follow these steps:

1.  **Install Extensions:**
    *   [C/C++ Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) from Microsoft.
    *   [CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools) from Microsoft.
    *   [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) extension from Microsoft.
2.  **Select Python Interpreter:**
    *   Open the project folder (`OmniDSP`) in VS Code.
    *   Use the Command Palette (`Ctrl+Shift+P`) and run `Python: Select Interpreter`.
    *   Choose the Python interpreter associated with your project environment (e.g., the one in your `.venv` or `omnidsp-py` virtual environment).
3.  **Configure CMake Tools:**
    *   CMake Tools will likely activate automatically. If not, use the Command Palette to run `CMake: Configure`.
    *   Select a CMake "Kit". For Windows with Visual Studio installed, choose the appropriate kit, typically `Visual Studio Community 2022 Release - amd64` (or similar based on your VS version and architecture). This should match the compiler used during `pip install`.
    *   Wait for the configuration step to complete. You should see output in the "Output" panel (select "CMake" from the dropdown).
4.  **IntelliSense Configuration:**

    **Method 1: Using Environment Configurator for Intel Tools (Recommended)**
    *   Install the [Environment Configurator for Intel® Software Development Tools](https://marketplace.visualstudio.com/items?itemName=intel-corporation.intel-sw-dev-tools-env-config) VS Code extension.
    *   Open VS Code and ensure the Intel oneAPI Base Toolkit environment is activated using the extension. This extension simplifies environment configuration for Intel tools within VS Code.
    *   After activating the environment, CMake Tools should automatically configure the project with the correct environment variables, including necessary paths for oneMKL and the compiler.
    *   CMake Tools will likely activate automatically. If not, use the Command Palette to run `CMake: Configure`.
    *   Select a CMake "Kit". For Windows with Visual Studio installed, choose the appropriate kit, typically `Visual Studio Community 2022 Release - amd64` (or similar based on your VS version and architecture). This should match the compiler used during `pip install`.
    *   Wait for the configuration step to complete. You should see output in the "Output" panel (select "CMake" from the dropdown).

    **Method 2: Manual Configuration (Alternative)**
    *   After a successful CMake configuration, CMake Tools automatically tells the C/C++ extension about the include paths used by the build (for C++, pybind11 headers from `WorkspaceContent`, and the detected Python `Include` directory).
    *   This should automatically configure IntelliSense. Open `python/bindings.cpp` - the `#include` errors for `<pybind11/...>` and CPython headers (like `<Python.h>` or `<frameobject.h>`) should disappear.
    *   **Troubleshooting:** If IntelliSense errors related to Python headers (e.g., `cannot open source file "frameobject.h"`) persist *after* a successful CMake configure:
        1.  Find your Python include path by running this in your activated environment's terminal:
            ```bash
            python -c "import sysconfig; print(sysconfig.get_path('include'))"
            ```
        2.  Open the Command Palette (`Ctrl+Shift+P`), run `C/C++: Edit Configurations (JSON)`.
        3.  Find your active configuration (e.g., "windows-msvc-x64").
        4.  Add the path obtained from step (a) to the `"includePath"` array.
        5.  Save the `c_cpp_properties.json` file. Reload the VS Code window if needed.
5.  **Conditional Python Build:** The root `CMakeLists.txt` uses an `OPTION` (`OMNIDSP_BUILD_PYTHON_BINDINGS`, default `OFF`) to control whether the Python bindings (`python/` subdirectory) are configured and built.
    *   For normal C++ development and IntelliSense in VS Code, this option stays `OFF`. CMake Tools configures based on this, but *should* still find pybind11/Python include paths if they are processed unconditionally before the `if()` block in the root `CMakeLists.txt` (as recommended).
    *   When building the Python package via `pip install .`, `pyproject.toml` tells `scikit-build-core` to pass `-DOMNIDSP_BUILD_PYTHON_BINDINGS=ON` to CMake, ensuring the Python module is built.

## Building from Source
# OmniDSP - High-Performance Cross-Platform DSP Library

## Overview

This project provides a high-performance C++ library for Digital Signal Processing (DSP). Its primary goal is to offer a consistent, easy-to-use API while leveraging highly optimized, platform-specific backends for maximum performance across various systems. It currently includes Fast Fourier Transform (FFT) and Constant-Q Transform (CQT) functionality, with plans to expand to a broader range of DSP functions. It automatically selects between Intel's oneAPI Math Kernel Library (oneMKL) or Apple's Accelerate framework where available to utilize hardware acceleration.

If neither backend is detected, a stub implementation is compiled which will throw runtime errors upon use, indicating that a functional backend was not available during the build.

## Features

*   **Cross-Platform Performance:** Designed for high-performance DSP across platforms, utilizing optimized backends.
*   **Platform Abstraction:** Single C++ API for DSP operations, simplifying development.
*   **Optimized Backends:** Uses Intel oneMKL (via oneAPI) or Apple Accelerate framework for hardware-accelerated FFTs.
*   **Automatic Backend Selection:** CMake build system detects and selects the appropriate backend during the build process.
*   **Python Bindings:** Provides Python bindings via pybind11 for easy integration with NumPy and the Python ecosystem.
*   **Precision Support:** Works with `float` (single) and `double` (double) precision values.
*   **Transform Types:** Supports Complex-to-Complex (C2C) and Real-to-Complex/Complex-to-Real (R2C/C2R) FFTs.
*   **Execution Modes:** Supports in-place and out-of-place execution for C2C transforms.
*   **Configurable Normalization:** Supports different scaling conventions for FFTs (`NormMode`).
*   **Plan-Based Interface:** `FFTPlan` class allows pre-computation for efficient execution of multiple FFTs.
*   **Convenience Functions:** Provides simple functions for common FFT operations using `std::vector`.
*   **Window Functions:** Includes implementations for common window functions like Hann, Hamming, Kaiser, and Flat-top.
*   **Build System:** Uses CMake for cross-platform building and dependency management.
*   **Installation Support:** Provides CMake installation rules and package configuration files for easy integration into other CMake projects.
*   **Unit Testing:** Includes a test suite using GoogleTest.

## Project Structure

```
OmniDSP/
│
├── CMakeLists.txt          # Main CMake build script
├── Config.cmake.in         # Template for CMake package config file
│
├── examples/               # Usage examples
│   ├── cpp/                # C++ examples
│   │   ├── CMakeLists.txt  # CMake script for C++ examples
│   │   ├── cqt.cpp
│   │   └── ...
│   └── python/             # Python examples
│       ├── requirements.txt
│       ├── run_fft_example.py
│       └── ...
│
├── include/                # Public header files
│   └── OmniDSP/
│       ├── omnidsp.h         # Main header (include )
│       ├── cqt.h             # CQT specific header
│       └── windows.h         # Windowing functions header
│
├── src/                    # Implementation source files
│   ├── omnidsp.cpp           # Platform-independent convenience functions
│   ├── cqt.cpp               # CQTPlan implementation using FFTPlan
│   ├── bindings.cpp          # Python bindings (pybind11) source
│   └── backend/                # Backend-specific implementations
│       ├── onemkl.cpp          # oneMKL specific implementation
│       ├── accelerate.cpp      # Apple Accelerate specific implementation
│       └── stub.cpp            # Fallback/error implementation
│
├── tests/                  # Unit tests
│   ├── CMakeLists.txt
│   └── *.cpp
│
├── build/                  # (Generated Directory) Build artifacts
│
└── README.md               # This file (or its source)
```

## Prerequisites

1.  **CMake:** Version 3.15 or later (3.21+ recommended).
2.  **C++ Compiler:** A compiler supporting C++17 (e.g., GCC, Clang, MSVC).
3.  **FFT Backend (at least one for full FFT functionality):**
    *   **Intel oneMKL:** Install Intel oneAPI Base Toolkit. Required if building/running on Windows/Linux, or macOS with MKL preference.
    *   **Apple Accelerate:** On macOS, included with Xcode and Command Line Tools. Used by default on macOS unless MKL is preferred.
4.  **Python:** Version 3.x (for Python bindings and examples).
5.  **Python Packages:** `numpy` is required at runtime to use the Python bindings and examples. Other packages like `matplotlib` might be used in specific examples. See `examples/python/requirements.txt`.
6.  **Git:** Required by CMake's `FetchContent` to download GoogleTest and pybind11.

## Development Environment Setup (VS Code)

For the best C++ development experience with IntelliSense (autocompletion, error checking, code navigation) in Visual Studio Code, follow these steps:

1.  **Install Extensions:**
    *   [C/C++ Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) from Microsoft.
    *   [CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools) from Microsoft.
    *   [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) extension from Microsoft.
2.  **Select Python Interpreter:**
    *   Open the project folder (`OmniDSP`) in VS Code.
    *   Use the Command Palette (`Ctrl+Shift+P`) and run `Python: Select Interpreter`.
    *   Choose the Python interpreter associated with your project environment (e.g., the one in your `.venv` or `omnidsp-py` virtual environment).
3.  **Configure CMake Tools:**
    *   CMake Tools will likely activate automatically. If not, use the Command Palette to run `CMake: Configure`.
    *   Select a CMake "Kit". For Windows with Visual Studio installed, choose the appropriate kit, typically `Visual Studio Community 2022 Release - amd64` (or similar based on your VS version and architecture). This should match the compiler used during `pip install`.
    *   Wait for the configuration step to complete. You should see output in the "Output" panel (select "CMake" from the dropdown).
4.  **IntelliSense Configuration:**
    *   After a successful CMake configuration, CMake Tools automatically tells the C/C++ extension about the include paths used by the build (for C++, pybind11 headers from `WorkspaceContent`, and the detected Python `Include` directory).
    *   This should automatically configure IntelliSense. Open `python/bindings.cpp` - the `#include` errors for `<pybind11/...>` and CPython headers (like `<Python.h>` or `<frameobject.h>`) should disappear.
    *   **Troubleshooting:** If IntelliSense errors related to Python headers (e.g., `cannot open source file "frameobject.h"`) persist *after* a successful CMake configure:
        1.  Find your Python include path by running this in your activated environment's terminal:
            ```bash
            python -c "import sysconfig; print(sysconfig.get_path('include'))"
            ```
        2.  Open the Command Palette (`Ctrl+Shift+P`), run `C/C++: Edit Configurations (JSON)`.
        3.  Find your active configuration (e.g., "windows-msvc-x64").
        4.  Add the path obtained from step (a) to the `"includePath"` array.
        5.  Save the `c_cpp_properties.json` file. Reload the VS Code window if needed.
5.  **Conditional Python Build:** The root `CMakeLists.txt` uses an `OPTION` (`OMNIDSP_BUILD_PYTHON_BINDINGS`, default `OFF`) to control whether the Python bindings (`python/` subdirectory) are configured and built.
    *   For normal C++ development and IntelliSense in VS Code, this option stays `OFF`. CMake Tools configures based on this, but *should* still find pybind11/Python include paths if they are processed unconditionally before the `if()` block in the root `CMakeLists.txt` (as recommended).
    *   When building the Python package via `pip install .`, `pyproject.toml` tells `scikit-build-core` to pass `-DOMNIDSP_BUILD_PYTHON_BINDINGS=ON` to CMake, ensuring the Python module is built.

## Building from Source

This section describes how to build the C++ library, the Python module, and the included examples directly from the source code repository using CMake for development or C++ usage.

1.  **Clone or download the source code:**
    ```bash
    # Example using git:
    git clone https://github.com/m-wells/OmniDSP.git
    cd OmniDSP
    ```
2.  **(Required for MKL Backend): Activate oneAPI Environment:** If you intend to build with the MKL backend (required on Windows/Linux, optional on macOS), ensure the oneAPI environment is configured in your terminal session **before** running CMake. This usually involves running the environment script from your oneAPI installation.
    ```bash
    # Linux/macOS example:
    # Replace with the actual path and script name for your installation
    source /path/to/intel/oneapi/setvars.sh

    # Windows example:
    # Replace with the actual path and script name for your installation (e.g., oneapi-vars.bat)
    "C:\Program Files (x86)\Intel\oneAPI\<version>\oneapi-vars.bat"
    ```
    Refer to Intel oneAPI documentation for the exact script location and name for your version.
3.  **Create a build directory:**
    ```bash
    mkdir build
    cd build
    ```
4.  **Configure using CMake:**
    ```bash
    # CMake will attempt to find Accelerate (on macOS) or oneMKL
    # (Ensure MKL environment is set first if targeting MKL)
    # OMNIDSP_BUILD_PYTHON_BINDINGS defaults to OFF for C++ builds
    cmake ..

    # Optional: To prefer MKL even if Accelerate is found (e.g., on macOS):
    # cmake .. -DFFT_PREFER_ONEMKL=ON

    # Optional: Build static libraries instead of shared
    # cmake .. -DBUILD_SHARED_LIBS=OFF

    # Optional: Explicitly enable Python bindings build (requires Python/pybind11)
    # cmake .. -DOMNIDSP_BUILD_PYTHON_BINDINGS=ON
    ```
    Review CMake output for backend selection (MKL or Accelerate) and confirmation that dependencies were found.
5.  **Compile the code:**
    ```bash
    # For single-configuration generators (Makefiles, Ninja):
    cmake --build . --parallel

    # For multi-configuration generators (e.g., Visual Studio):
    cmake --build . --config Release --parallel
    # Or: cmake --build . --config Debug --parallel
    ```
    This will build the C++ library (`libomnidsp`), C++ examples, and tests. It will only build the Python module if `-DOMNIDSP_BUILD_PYTHON_BINDINGS=ON` was passed during configuration.
6.  **Run Tests (Optional):**
    ```bash
    # From the build directory
    # Ensure MKL environment is active if tests require MKL runtime DLLs
    ctest -C Release --output-on-failure
    # Or: ctest -C Debug --output-on-failure
    ```

## Installation (Python Package via Pip)

The recommended way to install the Python package for usage in Python projects is via `pip`:

1.  Ensure prerequisites are met (CMake, C++ Compiler, oneAPI/MKL if needed, Git).
2.  **(Required for MKL Backend): Activate oneAPI Environment:** Open a terminal and activate the oneAPI environment using the appropriate script (e.g., `oneapi-vars.bat` on Windows) as shown in the "Building from Source" section. This is crucial so `scikit-build-core`/CMake can find MKL during the build.
3.  Navigate to the project's root directory (containing `pyproject.toml`).
4.  Run `pip install`:
    ```bash
    pip install .
    ```
    For troubleshooting, use verbose mode:
    ```bash
    pip install . -v
    ```

This command uses `scikit-build-core` (specified in `pyproject.toml`) to:
*   Invoke CMake with `OMNIDSP_BUILD_PYTHON_BINDINGS=ON`.
*   Compile the C++ library and the Python bindings.
*   Package the necessary components (Python module `.pyd`/`.so`, potentially bundled MKL DLLs if found by CMake during install) into a Python wheel.
*   Install the wheel into your Python environment's `site-packages`.

## Installation (C++ Library via CMake)

To install only the C++ library and headers (e.g., for use by other C++ projects) without the Python bindings:

1.  **Choose Installation Prefix (Optional but Recommended):** When configuring with CMake (in a clean build directory), specify where to install using `CMAKE_INSTALL_PREFIX`. If omitted, CMake uses a default system location which might require administrator privileges. Ensure `OMNIDSP_BUILD_PYTHON_BINDINGS` is `OFF` (default).
    ```bash
    # Example: Install to a 'staging' directory inside 'build'
    cmake -DCMAKE_INSTALL_PREFIX=./staging ..

    # Example: Install to a specific user location (Linux/macOS)
    # cmake -DCMAKE_INSTALL_PREFIX=~/local ..
    ```
2.  **Build the Library:**
    ```bash
    cmake --build . --config Release
    ```
3.  **Install the Library:**
    ```bash
    cmake --install . --config Release
    ```
    This command copies files to the location specified by `CMAKE_INSTALL_PREFIX`. You might need `sudo` (Linux/macOS) or administrator rights (Windows) if installing to a system-protected location.

After installation, the following structure should exist under your chosen `<prefix>`:
*   `<prefix>/include/OmniDSP/` (C++ headers)
*   `<prefix>/lib/` (C++ static/shared libraries, `.a`, `.lib`, `.so`)
*   `<prefix>/bin/` (`.dll` files on Windows)
*   `<prefix>/lib/cmake/OmniDSP/` (CMake package configuration files)
*   `<prefix>/share/OmniDSP/examples_python/` (Optional, Python examples are installed by default but could be made conditional)

## Using the Installed Library

### From C++
Once OmniDSP is installed, other CMake projects can find and use it.
1.  **CMakeLists.txt for Consuming Project:** In the `CMakeLists.txt` of your application:
    ```cmake
    cmake_minimum_required(VERSION 3.15)
    project(MyCoolApp LANGUAGES CXX)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # Find the installed OmniDSP package
    # If installed to a non-standard location, set CMAKE_PREFIX_PATH
    # e.g., list(APPEND CMAKE_PREFIX_PATH "/path/to/omnidsp/install")
    find_package(OmniDSP 1.0.0 REQUIRED)

    add_executable(my_app main.cpp)

    # Link against the imported OmniDSP target
    target_link_libraries(my_app PRIVATE OmniDSP::omnidsp)
    ```
2.  **C++ Code in Consuming Project:** Include the header and use the classes/functions:
    ```cpp
    #include <OmniDSP/omnidsp.h> // Use path relative to include directory
    #include <vector>
    #include <complex>
    #include <iostream>

    int main() {
        size_t N = 32;
        using Real = float;
        using Complex = std::complex<Real>;

        std::vector<Real> my_real_signal(N, 0.0f);
        my_real_signal[1] = 1.0f; // Simple delta function

        std::vector<Complex> my_spectrum; // Size N/2 + 1

        try {
            // Use the installed library's convenience function (rfft)
            OmniDSP::rfft(my_real_signal, my_spectrum);

            std::cout << "Spectrum calculated using installed OmniDSP library:" << std::endl;
            // ... print results ...

        } catch(const std::exception& e) {
            std::cerr << "OmniDSP Error: " << e.what() << std::endl;
            return 1;
        }

        return 0;
    }
    ```

### From Python

Once Omni DSP is installed using `pip install .`, you can use the Python module (`omnidsp_py`).

1.  **Ensure Module is Found by Python:**
    *   If you installed using `pip install .` into your active environment, it should be found automatically.
2.  **IMPORTANT: Ensure Runtime Dependencies are Found (Especially MKL):**
    The Python module depends on the core C++ library, which depends on its backend (MKL or Accelerate). If the MKL backend was used (common on Windows/Linux), the OS needs to find the MKL runtime libraries (e.g., `mkl_rt.dll`).

    **If you encounter an `ImportError: DLL load failed... The specified module could not be found`** (or similar), configure your runtime environment so the MKL runtime is found. Choose **one**:
    *   **Environment Setup Before Launch (Recommended):**
        *   Activate the Intel oneAPI environment in your terminal **before** launching Python/Jupyter/VS Code using the appropriate script (e.g., `oneapi-vars.bat` for your version on Windows, `source setvars.sh` on Linux/macOS).
        *   Alternatively, manually add the MKL runtime directory (e.g., `<ONEAPI_ROOT>/mkl/latest/bin/intel64` on Windows, `<ONEAPI_ROOT>/mkl/latest/lib/` on Linux/macOS) to your system's `PATH` (Windows) or `LD_LIBRARY_PATH` (Linux) / `DYLD_LIBRARY_PATH` (macOS).
    *   **`.env` File (IDE-Friendly / VS Code):**
        *   Create `.env` in your workspace root. Add a line prepending the MKL runtime directory to the appropriate variable (`PATH`, `LD_LIBRARY_PATH`, or `DYLD_LIBRARY_PATH`). See original README section for examples.
        *   VS Code's Python extension often loads this automatically. Reload VS Code if needed.
    *   **Code-Based (Windows Python 3.8+ Only):**
        *   Use `os.add_dll_directory()` in Python *before* the import.
    **Note:** This setup is generally not needed if using the Accelerate backend on macOS.
3.  **Import and Use:** Import the module and NumPy:
    ```python
    import numpy as np
    import os # For potential DLL path addition

    # --- Optional: Add MKL runtime path if needed (Windows Python 3.8+) ---
    # mkl_runtime_path = r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin\intel64"
    # if os.name == 'nt' and hasattr(os, 'add_dll_directory') and os.path.exists(mkl_runtime_path):
    #     try:
    #         os.add_dll_directory(mkl_runtime_path)
    #         print(f"Added MKL path: {mkl_runtime_path}")
    #     except Exception as e:
    #         print(f"Error adding MKL path: {e}")
    # # --- End Optional Path Addition ---

# Use the name you configured (e.g., omnidsp_py)
    import omnidsp_py

    # Example using rfft (double precision)
    N = 16
    signal = np.cos(2.0 * np.pi * 3.0 * np.arange(N) / N).astype(np.float64)

    try:
        # Use the bound convenience function
        spectrum = omnidsp_py.rfft_double(signal) # Use updated module name

        print(f"Input signal shape: {signal.shape}")
        print(f"Output spectrum shape: {spectrum.shape}") # Should be N/2 + 1
        print("Spectrum (first 5 bins):")
        print(spectrum[:5])

        # Example using CQTPlan (assuming appropriate setup)
        # sample_rate = 44100.0
        # window_func = ... # Define or import appropriate window function for Python side if needed
        # cqt_plan = omnidsp_py.CQTPlanDouble(sample_rate, ..., window_func) # Use updated module name
        # cqt_output = cqt_plan.execute(signal)
        # print(f"CQT Output shape: {cqt_output.shape}")

    except ImportError as e:
         print(f"ImportError: {e}")
         print("Ensure omnidsp_py module is installed correctly and MKL runtime (if used) is accessible.")
    except Exception as e:
        print(f"An error occurred: {e}")
    ```
4.  **See Examples:** Refer to the scripts in the `examples/python/` directory for more detailed usage.

## Running Examples

### C++ Examples

1.  Build the project using standard CMake commands as described in "Building from Source". Ensure `OMNIDSP_BUILD_PYTHON_BINDINGS` is `OFF` (default).
2.  The C++ example executables will be located in your build directory, typically under `build/examples/cpp/<Config>/` (e.g., `build/examples/cpp/Release/`).
3.  Run them from the build directory:
    ```bash
    # Example running from the build directory:
    ./examples/cpp/Release/fft # Adjust path as needed
    ./examples/cpp/Release/cqt
    ./examples/cpp/Release/rfft
    ./examples/cpp/Release/windows
    ```

### Python Examples

1.  Install the Python package using `pip install .` as described in "Installation (Python Package via Pip)".
2.  Navigate to the Python examples directory:
    ```bash
    cd examples/python # Run from the OmniDSP root directory
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt # Or: pip install numpy matplotlib librosa
    ```
4.  **Ensure Runtime Dependencies (MKL if used) can be found by Python.** See the "Using the Installed Library -> From Python -> IMPORTANT: Ensure Runtime Dependencies..." section above for environment setup options (e.g., activate oneAPI environment, set PATH/LD_LIBRARY_PATH, use `.env`).
5.  Run the desired example script:
    ```bash
    python run_fft_example.py
    ```

## Backend Selection (Build-time of OmniDSP)
*   By default, on Apple platforms (macOS), CMake will try to find and use the **Accelerate** framework when building OmniDSP.
*   On other platforms (Windows, Linux), or if Accelerate is not found/preferred on Apple, CMake will try to find **Intel oneMKL**.
*   You can force CMake to prefer oneMKL when building OmniDSP by setting the CMake option `FFT_PREFER_ONEMKL` to `ON`.
*   If neither Accelerate nor oneMKL is found when building OmniDSP, the library compiles using a **stub implementation**. Attempting to use the installed library will result in a `std::runtime_error`.
*   The choice of backend is fixed when OmniDSP is compiled; consuming applications use the backend that the installed library was built with.

## A Note on SYCL and GPU Acceleration
Users familiar with Intel oneAPI might wonder why the oneMKL backend uses the classic C-style DFTI API instead of the newer SYCL API. While SYCL offers potential performance benefits for very large FFTs on compatible GPU hardware, the current implementation uses the classic DFTI API for cross-platform CPU performance focus, reduced complexity, API consistency, and efficiency for small-to-medium FFT sizes common in many CPU-bound applications. Support for SYCL could be considered in the future.

## Notes / Limitations
*   **Inverse FFT Scaling & Normalization:** The library supports different normalization modes (`NormMode::BACKWARD`, `ORTHO`, `FORWARD`) via the `FFTPlan` constructor. The convenience functions (`fft`, `ifft`, `rfft`, `irfft`) currently use the default `NormMode::BACKWARD`. Be aware of the scaling factors associated with each mode and backend. The Python convenience functions also use `NormMode::BACKWARD`. Use `FFTPlanFloat/Double` via Python for other modes.
*   **Hermitian Symmetry for IRFFT:** The input to `irfft` (or `FFTPlan::execute_irfft`) must possess Hermitian symmetry for the resulting output signal to be purely real.
*   **Accelerate Real FFT Length:** The current implementation using Accelerate for the `Domain::REAL` transforms requires the real signal length `N` to be a **power of 2** (or N=1). Non-power-of-2 lengths will cause an error during plan creation if the Accelerate backend is used. oneMKL does not have this restriction.
*   **Thread Safety:** Using different `FFTPlan` objects from different threads is generally safe. Executing the *same committed plan* concurrently on *different data buffers* should be safe for both backends (consult backend docs for guarantees). Modifying the same `FFTPlan` object concurrently is unsafe.
*   **Supported Transforms:** Currently implements 1D C2C, R2C/C2R FFTs, and CQT. Multi-dimensional transforms are not supported. In-place execution is only explicitly provided for C2C transforms.
*   **Dependencies for Consumers:** Applications using OmniDSP (via C++ or Python) will implicitly depend on the runtime components of the backend OmniDSP was built with (e.g., oneMKL runtime libraries or the Accelerate framework). Ensure these are available on the system where the consuming application will run, following the environment setup steps if necessary.

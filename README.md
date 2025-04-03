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
*   **Configurable Normalization:** Supports different scaling conventions for FFTs.
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
├── examples/               # Usage examples
│   ├── cpp/                  # C++ examples
│   │   ├── CMakeLists.txt      # CMake script for C++ examples
│   │   ├── *.cpp
│   │   └── ...
│   └── python/               # Python examples
│       ├── requirements.txt
│       ├── *.py
│       └── ...
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
    *   **Intel oneMKL:** Install Intel oneAPI Base Toolkit.
    *   **Apple Accelerate:** On macOS, included with Xcode and Command Line Tools.
4.  **Python:** Version 3.x (for Python bindings and examples).
5.  **Python Packages:** `numpy` is required at runtime to use the Python bindings and examples. Other packages like `matplotlib` might be used in specific examples.
6.  **Git:** Required by CMake's `FetchContent` to download GoogleTest and pybind11.

## Building from Source

This section describes how to build the C++ library, the Python module, and the included examples directly from the source code repository.

1.  **Clone or download the source code:**
    
    ```bash
    # Example using git:
    git clone https://github.com/m-wells/OmniDSP.git
    cd OmniDSP
    ```
    
2.  **Create a build directory:**
    
    ```bash
    mkdir build
    cd build
    ```
    
3.  **Configure using CMake:**
    
    ```bash
    # CMake will attempt to find Accelerate (on macOS) or oneMKL
    cmake ..
    
    # Optional: To prefer MKL even if Accelerate is found (e.g., on macOS):
    # cmake .. -DFFT_PREFER_ONEMKL=ON
    
    # Optional: Build static libraries instead of shared (useful for testing)
    # cmake .. -DBUILD_SHARED_LIBS=OFF
    ```
    
    Review CMake output for backend selection and confirmation that pybind11 was found.
    
4.  **Compile the code:**
    
    ```bash
    # For single-configuration generators (Makefiles, Ninja):
    cmake --build .
    
    # For multi-configuration generators (e.g., Visual Studio):
    cmake --build . --config Release
    # Or: cmake --build . --config Debug
    ```

    This will build the C++ library (`libomnidsp`), the Python module (`omnidsp_py.***.so` or `.pyd`), and the C++ example executables.
    
5.  **Run Tests (Optional):**
    
    ```bash
    # From the build directory
    ctest -C Release --output-on-failure
    # Or: ctest -C Debug --output-on-failure
    ```
    
6.  **Run the example (optional):**
    
    The compiled example executable (`omnidsp_example` or `omnidsp_example.exe`) will be located within the `build` directory (potentially under a `Release` or `Debug` subdirectory).
    
    ```bash
    # Example:
    ./examples/Release/omnidsp_example # Adjust path based on generator/OS
    ```

    See the "Running Examples" section below for how to run the built examples.

## Installation

To install the OmniDSP C++ library, the Python module, and optionally the examples so they can be used system-wide or by other projects:

1.  **Choose Installation Prefix (Optional but Recommended):**
    
    When configuring with CMake, specify where to install using `CMAKE_INSTALL_PREFIX`. If omitted, CMake uses a default system location which might require administrator privileges for the install step.
    
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
*   `<prefix>/lib/python/` (or similar, containing `omnidsp_py.***.so`/`.pyd` module - exact path might vary based on CMake/pybind11 config)
*   `<prefix>/share/OmniDSP/examples_python/` (Optional, if Python examples installed)

## Using the Installed Library

### From C++

Once OmniDSP is installed, other CMake projects can find and use it.

1.  **CMakeLists.txt for Consuming Project:**
    
    In the `CMakeLists.txt` of your application:
    
    ```
    cmake_minimum_required(VERSION 3.15)
    project(MyCoolApp LANGUAGES CXX)
    
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    
    # Find the installed OmniDSP package
    # If installed to a non-standard location, set CMAKE_PREFIX_PATH
    find_package(OmniDSP 1.0.0 REQUIRED)
    
    add_executable(my_app main.cpp)
    
    # Link against the imported OmniDSP target
    target_link_libraries(my_app PRIVATE OmniDSP::omnidsp)
    ```
    
2.  **C++ Code in Consuming Project:**
    
    Include the header and use the classes/functions:
    
    ```
    #include <OmniDSP/omnidsp.h> // Use path relative to include directory
    #include <vector>
    #include <complex>
    #include <iostream>
    
    int main() {
        size_t N = 32; // Example size (Power of 2 for Accelerate REAL)
        using Real = float;
        using Complex = std::complex<Real>;

        std::vector<Real> my_real_signal(N, 0.0f);
        my_real_signal[1] = 1.0f; // Simple delta function

        std::vector<Complex> my_spectrum; // Size N/2 + 1

        try {
            // Use the installed OmniDSP library's convenience function (rfft)
            // This uses the default NormMode::BACKWARD normalization
            OmniDSP::rfft(my_real_signal, my_spectrum);

            std::cout << "Spectrum calculated using installed OmniDSP library:" << std::endl;
            for(size_t i = 0; i < my_spectrum.size(); ++i) {
                std::cout << i << ": " << my_spectrum[i] << std::endl;
            }

            // Example using a plan with ORTHO normalization
            OmniDSP::FFTPlan<Real> plan(N, OmniDSP::Precision::SINGLE,
                                      OmniDSP::Direction::FORWARD, OmniDSP::Domain::REAL,
                                      OmniDSP::NormMode::ORTHO);
            std::vector<Complex> spectrum_ortho(plan.getComplexLength());
            plan.execute_rfft(my_real_signal.data(), spectrum_ortho.data());
            std::cout << "\nOrtho Spectrum[0]: " << spectrum_ortho[0] << std::endl;


        } catch(const std::exception& e) {
            std::cerr << "OmniDSP Error: " << e.what() << std::endl;
            return 1;
        }

        return 0;
    }
    ```

### From Python

Once OmniDSP is built (and optionally installed), you can use the Python module.

1.  **Ensure Module is Found:**
    *   If you installed OmniDSP to a standard Python location (e.g., site-packages), it should be found automatically.
    *   If you installed to a custom prefix, you might need to add the Python module's directory (e.g., `<prefix>/lib/python`) to your `PYTHONPATH` environment variable.
    *   If running directly from the build directory, add the directory containing the built `omnidsp_py***.so/.pyd` file (usually inside `build/`) to your `PYTHONPATH`.
2.  **Import and Use:**
    
    Import the module and necessary libraries (like NumPy):
    
    ```
    import numpy as np
    import omnidsp_py # The name defined in bindings.cpp/CMakeLists.txt
    
    # Example using rfft (double precision)
    N = 16
    signal = np.cos(2.0 * np.pi * 3.0 * np.arange(N) / N).astype(np.float64)
    
    try:
        # Use the bound convenience function
        spectrum = omnidsp_py.rfft_double(signal)
    
        print(f"Input signal shape: {signal.shape}")
        print(f"Output spectrum shape: {spectrum.shape}") # Should be N/2 + 1
        print("Spectrum (first 5 bins):")
        print(spectrum[:5])
    
        # Example using CQTPlan (assuming appropriate setup)
        # sample_rate = 44100.0
        # cqt_plan = omnidsp_py.CQTPlanDouble(...)
        # cqt_output = cqt_plan.execute(signal)
        # print(f"CQT Output shape: {cqt_output.shape}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    ```
    
3.  **See Examples:** Refer to the scripts in the `examples/python/` directory for more detailed usage.

## Running Examples

### C++ Examples

1.  Build the project as described in "Building from Source".
2.  The C++ example executables will be located in your build directory, typically under `examples/cpp/` (possibly within a configuration subfolder like `Release` or `Debug` depending on your CMake generator).
    
    ```
    # Example running from the build directory:
    ./examples/cpp/Release/fft # Adjust path as needed
    ./examples/cpp/Release/cqt
    ```
    

### Python Examples

1.  Build the project as described in "Building from Source". This creates the `omnidsp_py` module.
2.  Navigate to the Python examples directory:
    
    ```
    cd examples/python # Run from the OmniDSP root directory
    # Or navigate to the installed location if you installed examples
    ```
    
3.  Install dependencies (primarily NumPy):
    
    ```
    pip install -r requirements.txt # Or: pip install numpy
    ```
    
4.  Ensure the `omnidsp_py` module can be found by Python. Adjust the path to point to the directory containing the built `omnidsp_py***.so` or `.pyd` file. If you installed OmniDSP system-wide or to your Python environment's site-packages, this step might not be necessary.
    * Example for Linux/macOS, assuming build dir is `../build` relative to OmniDSP root
        ```
        export PYTHONPATH=$PYTHONPATH:$(pwd)/../build
        ```
    * Example for Windows (Command Prompt)
        ```
        set PYTHONPATH=%PYTHONPATH%;C:\path\to\OmniDSP\build
        ```
    * Example for Windows (PowerShell)
        ```
        $env:PYTHONPATH += ";C:\path\to\OmniDSP\build"
        ```
    
5.  Run the desired example script:
    
    ```
    python <example>.py
    ```

## Backend Selection (Build-time of OmniDSP)

*   By default, on Apple platforms (macOS), CMake will try to find and use the **Accelerate** framework when building OmniDSP.
*   On other platforms, or if Accelerate is not found/preferred on Apple, CMake will try to find **Intel oneMKL**.
*   You can force CMake to prefer oneMKL when building OmniDSP by setting the CMake option `FFT_PREFER_ONEMKL` to `ON`.
*   If neither Accelerate nor oneMKL is found when building OmniDSP, the library compiles using a **stub implementation**. Attempting to use the installed library will result in a `std::runtime_error`.
*   The choice of backend is fixed when OmniDSP is compiled; consuming applications use the backend that the installed library was built with.

## A Note on SYCL and GPU Acceleration

Users familiar with Intel oneAPI might wonder why the oneMKL backend uses the classic C-style DFTI API instead of the newer SYCL API, which can target Intel GPUs. While the SYCL API offers potential performance benefits for very large FFTs on compatible GPU hardware, the current OmniDSP implementation uses the classic DFTI API for the following reasons:

*   **Cross-Platform CPU Performance Focus:** The primary goal of OmniDSP is to provide a unified way to achieve high performance DSP across multiple platforms, primarily leveraging their CPU capabilities. The classic oneMKL DFTI provides excellent, highly optimized CPU performance that aligns well with the capabilities targeted by the Apple Accelerate backend, ensuring a more consistent performance profile across supported systems.
*   **Reduced Complexity:** Integrating SYCL requires the DPC++ compiler toolchain, adds asynchronous programming concepts (queues, events), and necessitates careful data management (SYCL buffers or USM) which significantly increases the build and implementation complexity compared to the direct DFTI approach.
*   **API Consistency & Overhead:** To maintain the current simple host-pointer API (e.g., `execute(const double*...)`), a SYCL backend would need to internally handle data transfers between the host CPU and the device (CPU or GPU). This transfer overhead can negate the GPU's computational advantage except for **very large FFT sizes (e.g., potentially hundreds of thousands or millions of points, depending heavily on the specific hardware)**. For smaller to medium sizes frequently encountered in CPU-bound applications, the data transfer time and kernel launch overhead often make the highly optimized classic CPU API faster and more efficient. The exact crossover point depends heavily on the specific CPU, GPU, memory bandwidth, data precision, and whether batching is used.
*   **Target Use Case:** The classic DFTI API is often sufficient or even faster for the small-to-medium FFT sizes common in many CPU-bound applications.

Therefore, the current approach prioritizes broad cross-platform CPU performance, implementation simplicity, and API accessibility. Support for SYCL could be considered in the future if targeting Intel GPU acceleration for very large problem sizes becomes a primary requirement for the library.

## Notes / Limitations

*   **Inverse FFT Scaling & Normalization:** The library supports different normalization modes (`NormMode::BACKWARD`, `ORTHO`, `FORWARD`) via the `FFTPlan` constructor. The convenience functions (`fft`, `ifft`, `rfft`, `irfft`) currently use the default `NormMode::BACKWARD`. Be aware of the scaling factors associated with each mode and backend:
    *   **oneMKL:** Scaling factors are set explicitly via the MKL API according to the chosen `NormMode`.
    *   **Accelerate:** Scaling is applied manually within the library after calling Accelerate functions to match the chosen `NormMode`. Verify results carefully, especially if comparing between backends or requiring strict numerical normalization.

     The Python convenience functions currently use `NormMode::BACKWARD`. Use `FFTPlanFloat/Double` via Python for other modes.**
*   **Hermitian Symmetry for IRFFT:** The input to `irfft` (or `FFTPlan::execute_irfft`) must possess [Hermitian symmetry](https://math.stackexchange.com/questions/141180/hermitian-property-for-discrete-fourier-transform) for the resulting output signal to be purely real. This symmetry is naturally produced by the forward real FFT (`rfft`). Providing non-Hermitian input may lead to non-real output or undefined behavior depending on the backend.
*   **Accelerate Real FFT Length:** The current implementation using Accelerate for the `Domain::REAL` transforms (`rfft`/`irfft` via `FFTPlan`) requires the real signal length `N` to be a **power of 2** (or N=1). Non-power-of-2 lengths will cause an error during plan creation if the Accelerate backend is used. oneMKL does not have this restriction.
*   **Thread Safety:**
    *   Using different `FFTPlan` objects from different threads is generally safe.
    *   Executing the _same committed plan_ (`FFTPlan::execute...`) concurrently on _different data buffers_ should be safe for both oneMKL and Accelerate backends, but consult their respective documentation for definitive guarantees.
    *   Modifying the same `FFTPlan` object concurrently is unsafe. The internal reconfiguration for in-place C2C execution in the oneMKL backend is not thread-safe if called concurrently on the _same_ `FFTPlan` object.
*   **Supported Transforms:** Currently implements 1D C2C, R2C/C2R FFTs, and CQT. Multi-dimensional transforms are not supported. In-place execution is only explicitly provided for C2C transforms.
*   **Dependencies for Consumers:** Applications using OmniDSP (via C++ or Python) will implicitly depend on the runtime components of the backend OmniDSP was built with (e.g., oneMKL runtime libraries or the Accelerate framework). Ensure these are available on the system where the consuming application will run.
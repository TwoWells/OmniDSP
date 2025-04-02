# OmniDSP - High-Performance Cross-Platform DSP Library

## Overview

This project provides a high-performance C++ library for Digital Signal Processing (DSP). Its primary goal is to offer a consistent, easy-to-use API while leveraging highly optimized, platform-specific backends for maximum performance across various systems. It currently includes Fast Fourier Transform (FFT) functionality, with plans to expand to a broader range of DSP functions. It automatically selects between Intel's oneAPI Math Kernel Library (oneMKL) or Apple's Accelerate framework where available to utilize hardware acceleration.

If neither backend is detected, a stub implementation is compiled which will throw runtime errors upon use, indicating that a functional backend was not available during the build.

## Features

*   **Cross-Platform Performance:** Designed for high-performance DSP across platforms, utilizing optimized backends.
*   **Platform Abstraction:** Single C++ API for DSP operations, simplifying development.
*   **Optimized Backends:** Uses Intel oneMKL (via oneAPI) or Apple Accelerate framework for hardware-accelerated FFTs.
*   **Automatic Backend Selection:** CMake build system detects and selects the appropriate backend during the build process.
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
├── include/                # Public header files
│   └── OmniDSP/
│       └── omnidsp.h       # Main header (include )
│
├── src/                    # Implementation source files
│   ├── omnidsp.cpp           # Platform-independent convenience functions
│   ├── fft_impl_onemkl.cpp   # oneMKL specific implementation
│   ├── fft_impl_accelerate.cpp # Apple Accelerate specific implementation
│   └── fft_impl_stub.cpp     # Fallback/error implementation
│
├── examples/               # Usage examples within this repo
│   └── main.cpp
│
├── tests/                  # Unit tests
│   ├── CMakeLists.txt
│   ├── fft.cpp
│   ├── ifft.cpp
│   ├── rfft.cpp
│   ├── irfft.cpp
│   └── window.cpp
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
4.  **Git:** Required by CMake's `FetchContent` to download GoogleTest.

## Building from Source

This section describes how to build the library and the included example directly from the source code repository. See the 'Installation' section for installing the library for use by other projects.

1.  **Clone or download the source code:**
    
    ```bash
    # Example using git:
    git clone https://github.com/m-wells/OmniDSP.git # Replace URL
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
    
    Review CMake output for backend selection.
    
4.  **Compile the code:**
    
    ```bash
    # For single-configuration generators (Makefiles, Ninja):
    cmake --build .
    
    # For multi-configuration generators (e.g., Visual Studio):
    cmake --build . --config Release
    # Or: cmake --build . --config Debug
    ```
    
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
    

## Installation

To install the OmniDSP library (header, library files, CMake configuration files) so that other CMake projects can easily find and use it, follow these steps after configuring (step 3 in the "Building from Source" section):

1.  **Choose Installation Prefix (Optional but Recommended):**
    
    When configuring with CMake, specify where to install using `CMAKE_INSTALL_PREFIX`. If omitted, CMake uses a default system location which might require administrator privileges for the install step.
    
    ```bash
    # Example: Install to a 'staging' directory inside 'build'
    cmake -DCMAKE_INSTALL_PREFIX=./staging ..
    
    # Example: Install to a specific user location (Linux/macOS)
    # cmake -DCMAKE_INSTALL_PREFIX=~/libs/omnidsp ..
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

*   `<prefix>/include/OmniDSP/omnidsp.h`
*   `<prefix>/lib/` (containing `.a`, `.lib`, `.so` files)
*   `<prefix>/bin/` (containing `.dll` files on Windows)
*   `<prefix>/lib/cmake/OmniDSP/` (containing CMake package files)

## Using the Installed Library

Once OmniDSP is installed, other CMake projects can find and use it.

### 1. CMakeLists.txt for Consuming Project:

In the `CMakeLists.txt` of your application:

```cmake
cmake_minimum_required(VERSION 3.15)
project(MyCoolApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the installed OmniDSP package
find_package(OmniDSP 1.0.0 REQUIRED)

add_executable(my_app main.cpp YourOtherFiles.cpp)

# Link against the imported OmniDSP target provided by find_package
# This automatically handles include directories and library dependencies
target_link_libraries(my_app PRIVATE OmniDSP::omnidsp)

```

**Note on Finding the Package:** If you installed OmniDSP to a custom location, users need to tell CMake where to find it when configuring _their application_ by setting the `CMAKE_PREFIX_PATH` variable:

```bash
# Example: OmniDSP installed to ~/libs/omnidsp
cmake -DCMAKE_PREFIX_PATH=~/libs/omnidsp /path/to/your/MyCoolApp/source
```

### 2. C++ Code in Consuming Project:

In your application's source code, include the header using the path relative to the install prefix's `include` directory:

```cpp
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
*   **Hermitian Symmetry for IRFFT:** The input to `irfft` (or `FFTPlan::execute_irfft`) must possess [Hermitian symmetry](https://math.stackexchange.com/questions/141180/hermitian-property-for-discrete-fourier-transform) for the resulting output signal to be purely real. This symmetry is naturally produced by the forward real FFT (`rfft`). Providing non-Hermitian input may lead to non-real output or undefined behavior depending on the backend.
*   **Accelerate Real FFT Length:** The current implementation using Accelerate for the `Domain::REAL` transforms (`rfft`/`irfft` via `FFTPlan`) requires the real signal length `N` to be a **power of 2** (or N=1). Non-power-of-2 lengths will cause an error during plan creation if the Accelerate backend is used. oneMKL does not have this restriction.
*   **Thread Safety:**
    *   Using different `FFTPlan` objects from different threads is generally safe.
    *   Executing the _same committed plan_ (`FFTPlan::execute...`) concurrently on _different data buffers_ should be safe for both oneMKL and Accelerate backends, but consult their respective documentation for definitive guarantees.
    *   Modifying the same `FFTPlan` object concurrently is unsafe. The internal reconfiguration for in-place C2C execution in the oneMKL backend is not thread-safe if called concurrently on the _same_ `FFTPlan` object.
*   **Supported Transforms:** Currently implements 1D Complex-to-Complex (C2C) and 1D Real <-> Complex (R2C/C2R) transforms. Multi-dimensional transforms are not supported. In-place execution is only explicitly provided for C2C transforms.
*   **Dependencies for Consumers:** Applications linking against the installed OmniDSP library will implicitly depend on the runtime components of the backend OmniDSP was built with (e.g., oneMKL runtime libraries or the Accelerate framework). Ensure these are available on the system where the consuming application will run.
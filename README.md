# Cross-Platform C++ FFT Library

## Overview

This project provides a C++ library for performing Fast Fourier Transforms (FFTs). Its primary goal is to offer a consistent, easy-to-use API while leveraging highly optimized, platform-specific backends for performance. It automatically selects between Intel's oneAPI Math Kernel Library (oneMKL) or Apple's Accelerate framework based on the target machine and build configuration.

It supports standard Complex-to-Complex (C2C) transforms as well as Real-to-Complex (R2C) and Complex-to-Real (C2R) transforms for real-valued signals.

If neither backend is detected, a stub implementation is compiled which will throw runtime errors upon use, indicating that a functional backend was not available during the build.

## Features

*   **Platform Abstraction:** Single C++ API for FFT operations.
*   **Optimized Backends:** Uses Intel oneMKL (via oneAPI) or Apple Accelerate framework where available.
*   **Automatic Backend Selection:** CMake build system detects and selects the appropriate backend.
*   **Precision Support:** Works with `float` (single) and `double` (double) precision values.
*   **Transform Types:** Supports:
    *   Complex-to-Complex (C2C): Forward (`FORWARD`) and Inverse (`INVERSE`).
    *   Real-to-Complex (R2C): Forward (`FORWARD`, `Domain::REAL`).
    *   Complex-to-Real (C2R): Inverse (`INVERSE`, `Domain::REAL`).
*   **Execution Modes:** Supports both out-of-place (separate input/output buffers) and in-place (input buffer is overwritten for C2C transforms). R2C/C2R currently implemented as out-of-place.
*   **Plan-Based Interface:** `FFTPlan` class allows pre-computation for efficient execution of multiple FFTs of the same size/type.
*   **Convenience Functions:** Simple `fft`, `ifft`, `fft_inplace`, `ifft_inplace` (for C2C) and `fft_r2c`, `ifft_c2r` (for R2C/C2R) functions using `std::vector`.
*   **Build System:** Uses CMake for cross-platform building and dependency management.

## Project Structure

```
<project_root>/
│
├── CMakeLists.txt          # Main CMake build script
│
├── include/                # Public header files
│   └── fft_lib.h
│
├── src/                    # Implementation source files
│   ├── fft_lib.cpp             # Platform-independent convenience functions
│   ├── fft_impl_onemkl.cpp   # oneMKL specific implementation
│   ├── fft_impl_accelerate.cpp # Apple Accelerate specific implementation
│   └── fft_impl_stub.cpp     # Fallback/error implementation
│
├── examples/               # Usage examples
│   └── main.cpp
│
├── build/                  # (Generated Directory) Build artifacts
│
└── README.md               # This file
```

## Prerequisites

1.  **CMake:** Version 3.15 or later.
2.  **C++ Compiler:** A compiler supporting C++17 (e.g., GCC, Clang, MSVC).
3.  **FFT Backend (at least one):**
    *   **Intel oneMKL:** Install Intel oneAPI Base Toolkit (which includes MKL). Ensure the MKL environment variables are set up correctly (e.g., by running the `setvars.sh` or `setvars.bat` script provided by oneAPI, or ensuring `MKLROOT` is set) so CMake's `find_package(MKL)` can locate it.
    *   **Apple Accelerate:** On macOS, this is typically included with Xcode and the Command Line Tools. Note that the Real FFT implementation using Accelerate currently requires the signal length (N) to be a power of 2.

## Building

This project uses a standard CMake out-of-source build process.

1.  **Clone or download the source code:**
    
    ```bash
    # Example using git:
    # git clone <repository_url>
    # cd <project_root>
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
    ```
    
    Review the CMake output to see which backend (`ACCELERATE`, `ONEMKL`, or `NONE`) was selected.
    
4.  **Compile the code:**
    
    ```bash
    # For single-configuration generators (Makefiles, Ninja):
    cmake --build .
    
    # For multi-configuration generators (e.g., Visual Studio):
    cmake --build . --config Release
    # Or: cmake --build . --config Debug
    ```
    
5.  **Run the example (optional):**
    
    The compiled example executable (`fft_example` or `fft_example.exe`) will be located within the `build` directory (potentially under a `Release` or `Debug` subdirectory).
    

## Usage

Include the library header and use the `CrossPlatformFFT` namespace.

### Complex-to-Complex Example:

```cpp
#include <vector>
#include <complex>
#include <iostream>
#include "fft_lib.h"

int main() {
    const size_t N = 16;
    using Complex = std::complex<double>;
    std::vector<Complex> signal_c2c(N);
    // ... fill signal_c2c ...
    std::vector<Complex> spectrum_c2c;

    try {
        // Use convenience function
        CrossPlatformFFT::fft(signal_c2c, spectrum_c2c);

        // Or use FFTPlan
        CrossPlatformFFT::FFTPlan<double> plan(N,
                                               CrossPlatformFFT::Precision::DOUBLE,
                                               CrossPlatformFFT::Direction::FORWARD,
                                               CrossPlatformFFT::Domain::COMPLEX);
        std::vector<Complex> spectrum_plan(N);
        plan.execute(signal_c2c.data(), spectrum_plan.data());

        // ... process spectrum ...

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

### Real-to-Complex / Complex-to-Real Example:

```cpp
#define _USE_MATH_DEFINES // Required for M_PI with MSVC <cmath>
#include <cmath>
#include <vector>
#include <complex>
#include <iostream>
#include "fft_lib.h"

int main() {
    const size_t N = 16; // Real signal length (must be power of 2 for Accelerate backend)
    using Complex = std::complex<double>;
    using Real = double;

    std::vector<Real> real_signal(N);
    for (size_t i = 0; i < N; ++i) {
        real_signal[i] = std::cos(2.0 * M_PI * 3.0 * i / N);
    }

    std::vector<Complex> complex_spectrum; // Size N/2 + 1
    std::vector<Real> reconstructed_signal; // Size N

    try {
        // R2C Transform (using convenience function)
        CrossPlatformFFT::fft_r2c(real_signal, complex_spectrum);
        std::cout << "R2C Spectrum (Size " << complex_spectrum.size() << "):" << std::endl;
        // ... print spectrum ...

        // C2R Transform (using convenience function)
        CrossPlatformFFT::ifft_c2r(complex_spectrum, reconstructed_signal);
        std::cout << "\nC2R Reconstructed (Size " << reconstructed_signal.size() << "):" << std::endl;
        // ... print reconstructed signal ...

        // Note: Manual scaling might be needed for reconstructed_signal
        // if using Accelerate backend. See Notes section.

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

## Backend Selection

*   By default, on Apple platforms (macOS), CMake will try to find and use the **Accelerate** framework.
*   On other platforms, or if Accelerate is not found/preferred on Apple, CMake will try to find **Intel oneMKL**.
*   You can force CMake to prefer oneMKL even if Accelerate is available by setting the CMake option `FFT_PREFER_ONEMKL` to `ON`:
    
    ```bash
    cmake .. -DFFT_PREFER_ONEMKL=ON
    ```
    
*   If neither Accelerate nor oneMKL is found, the library will compile using a **stub implementation**. Attempting to create an `FFTPlan` or use the convenience functions will result in a `std::runtime_error` being thrown, indicating that no functional backend was available. Check the CMake configuration output (`FFT_BACKEND` variable) to confirm which backend is being used.

## Notes / Limitations

*   **Inverse FFT Scaling:** Be aware of scaling conventions, especially for Complex-to-Real (C2R) transforms.
    *   **oneMKL:** Typically applies the correct `1/N` scaling factor automatically during inverse transforms (C2C and C2R).
    *   **Accelerate:** The vDSP routines used often require manual scaling management. The C2R implementation here attempts to apply necessary pre-scaling to achieve an effective `1/N` overall scaling, but verify against Accelerate documentation. The C2C inverse transform via the DFT API likely does not apply scaling automatically.
    *   The convenience functions (`ifft`, `ifft_c2r`) do _not_ automatically apply any scaling beyond what the chosen backend does by default or what the implementation applies internally (as in the Accelerate C2R case). Check results and apply manual scaling if needed.
*   **Accelerate Real FFT Length:** The current implementation using Accelerate for the `Domain::REAL` transforms (via `vDSP_create_fftsetup`) requires the real signal length `N` to be a **power of 2**. Non-power-of-2 lengths will cause an error during plan creation with the Accelerate backend. oneMKL does not have this restriction.
*   **Thread Safety:**
    *   Using different `FFTPlan` objects from different threads is safe.
    *   Executing the _same committed plan_ (`FFTPlan::execute...`) concurrently on _different data buffers_ is generally safe for both oneMKL and Accelerate backends (refer to specific backend documentation for guarantees).
    *   Modifying the same `FFTPlan` object concurrently (which shouldn't typically happen after creation) is unsafe. Specifically, the internal reconfiguration for in-place C2C execution in the oneMKL backend is not thread-safe if called concurrently on the _same_ `FFTPlan` object.
*   **Supported Transforms:** Currently only supports 1D Complex-to-Complex (C2C) and 1D Real <-> Complex (R2C/C2R) transforms. Multi-dimensional transforms are not implemented. R2C/C2R are currently only implemented out-of-place via the API.
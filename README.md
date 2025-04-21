# OmniDSP

OmniDSP is a C++ library designed for high-performance Digital Signal Processing tasks, with Python bindings provided via pybind11. It aims to offer efficient implementations of common DSP algorithms by abstracting optimized backend libraries like Intel oneMKL (including IPP) and Apple Accelerate.

**Core Design Philosophy:** OmniDSP centralizes backend management. Users create an `OmniDSP` object, which selects and initializes the appropriate backend (MKL or Accelerate) based on the build environment. This instance then serves as the primary interface for accessing all DSP functionalities. This approach makes the link between backend configuration and function execution explicit, ensuring clarity and control, rather than using standalone functions that might obscure their dependence on the backend state or setup. It also provides a foundation for flexibility, such as allowing multiple backend instances or _potential_ runtime switching in the _future_.

## Table of Contents

- [OmniDSP](#omnidsp)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Dependencies](#dependencies)
  - [Project Status](#project-status)
  - [Project Structure](#project-structure)
  - [Getting Started and Installation](#getting-started-and-installation)
  - [Basic Usage](#basic-usage)
    - [Python](#python)
    - [C++](#c)
  - [Contributing](#contributing)
  - [License](#license)

## Features

- **Centralized Backend Management:** The main `OmniDSP` class handles backend selection (Intel oneMKL/IPP or Apple Accelerate), initialization, and serves as the explicit context for all operations.
- **Plan-Based Operations:** Uses efficient, stateful `Plan` objects for operations requiring setup (FFT, CQT, Resampling). Plans are created via factory methods on an `OmniDSP` instance, ensuring they operate within the correct backend context.
  - **Fast Fourier Transforms (FFT):** `FFTPlan` and `RFFTPlan` for Complex-to-Complex, Real-to-Complex, and Complex-to-Real transforms with configurable normalization.
  - **Constant-Q Transform (CQT):** `CQTPlan` for efficient recursive CQT computation with configurable parameters.
  - **Resampling:** `ResamplePlan` for configurable FIR filtering and downsampling/upsampling.
- **Stateless Operations:** Common DSP operations accessed directly via methods on an `OmniDSP` instance, making their execution context clear.
  - **Convolution / Correlation:** 1D linear convolution and correlation (`valid` mode currently implemented).
  - **Window Coefficient Generation:** Methods to generate coefficients for standard windows (Hann, Hamming, Kaiser, Flattop).
- **Optimized Backends:** Leverages MKL/IPP or Accelerate/vDSP for performance, selected automatically at build time. Includes a stub backend for compatibility.
- **PIMPL Idiom:** Uses the Pointer-to-Implementation pattern extensively to hide backend details, ensure ABI stability, and improve compile times.
- **Python Bindings:** Easy-to-use Python API mirroring the C++ structure via pybind11, requiring an `OmniDSP` instance.

## Dependencies

OmniDSP requires the following to build and run:

- A C++17 compliant compiler (e.g., GCC, Clang, MSVC)
- CMake (version 3.21 or higher)
- Python (version 3.8 or higher, if building Python bindings)
- **Boost C++ Libraries:** Required for certain cross-platform mathematical functions (specifically, the Bessel function needed for Kaiser window generation).
- Optional: Intel oneMKL (including IPP) for the MKL backend (recommended on Linux/Windows for performance).
- Optional: macOS users automatically use the Accelerate framework backend if oneMKL is not preferred/available.

**Note:** All required dependencies (except for the C++ compiler and the macOS Accelerate framework) are managed via the provided Conda `environment.yml` file. Please see the Installation section for details on setting up the environment.

## Project Status

- Core FFT, CQT (recursive), Convolution/Correlation (valid mode), and Windowing functionalities are implemented within the new architecture.
- **Recent Refactoring:** The library structure has been refactored around a central `OmniDSP` class that manages the backend and acts as a factory for `Plan` objects (`FFTPlan`, `CQTPlan`, `ResamplePlan`). Standalone functions and the static `Window` class have been removed from the public API to promote clarity about the backend execution context.
- **Current Work:** Implementing `ResamplePlan`, updating CQT to use it, adding convolution modes, resolving any remaining test failures post-refactor, tuning CQT scaling, and setting up CI. See `TODO.md`.

## Project Structure

- `include/OmniDSP/`: Public C++ headers defining the API (`omnidsp.h`, `fft.h`, `cqt.h`, `resample.h`, `core_types.h`).
- `src/omnidsp/`: Core C++ library implementation files (`omnidsp.cpp`, `fft.cpp`, `cqt.cpp`, `resample.cpp`). These implement the public API classes and forward calls to the backend implementations via PIMPL.
- `src/omnidsp/backend/`: Contains backend-specific implementations and the internal interface header.
  - `backend.h`: Internal header defining implementation class interfaces/signatures (within `OmniDSP::backend` namespace). **Not part of the public API.**
  - `onemkl/`: Implementations using Intel oneMKL/IPP (namespace `OmniDSP::backend::oneMKL`).
  - `accelerate/`: Implementations using Apple Accelerate/vDSP (namespace `OmniDSP::backend::Accelerate`).
  - `stub/`: Stub implementations that throw runtime errors (namespace `OmniDSP::backend::Stub`).
- `src/omnidsp_py/`: Python bindings source code (pybind11 wrappers).
- `tests/`: Unit tests.
  - `cpp/`: C++ tests using GoogleTest.
  - `python/`: Python tests using pytest.
- `examples/`: Usage examples.
  - `cpp/`: C++ examples.
  - `notebooks/`: Python examples using Jupyter notebooks.
- `environment.yml`, `environment-dev.yml`, `conda-*.lock`: Conda environment files.
- `.pre-commit-config.yaml`: Configuration for developer Git hooks.
- `CMakeLists.txt`: Main CMake build script.
- `pyproject.toml`: Python packaging configuration.
- `TODO.md`: Current development tasks.
- `CONTRIBUTING.md`: Guidelines for contributors.

## Getting Started and Installation

Using OmniDSP requires **Conda** to manage dependencies.

1.  **Set up Conda Environment:**

    - Clone the repository.
    - Create and activate the environment using the provided lock files (recommended) or `environment.yml`. Choose the file matching your platform (e.g., `linux-64`, `osx-arm64`, `win-64`):
      ```bash
      # Example for Linux:
      conda create --name omnidsp --file conda-linux-64.lock
      conda activate omnidsp
      ```
    - **Important:** Always ensure the `omnidsp` Conda environment is activated before building or using the library.

2.  **Install OmniDSP Python Package:**

    - Once the `omnidsp` environment is activated, install the package from the project root directory:
      ```bash
      pip install . -v
      ```
    - This compiles the C++ library and Python bindings and installs them into your active Conda environment.

3.  **(Optional) Build C++ Library Only:** Refer to `CONTRIBUTING.md` for instructions using CMake directly.

## Basic Usage

### Python

```python
import numpy as np
import omnidsp_py as ods # Assuming this is the import name

# 1. Create an OmniDSP instance (manages the backend)
try:
    dsp = ods.OmniDSP()
except RuntimeError as e:
    print(f"Error creating OmniDSP instance (backend issue?): {e}")
    exit()

# --- FFT Example ---
fs = 1000
t = np.arange(fs) / fs
signal_in_real = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
signal_in_real = signal_in_real.astype(np.float32)

# Create an RFFT plan using the factory method on the dsp instance
try:
    # Specify length, precision via factory
    rfft_plan = dsp.create_rfft_plan(
        length=len(signal_in_real),
        precision=ods.Precision.SINGLE
    )
    # Execute the plan
    complex_spectrum = rfft_plan.execute(signal_in_real)
    print(f"RFFT Output shape: {complex_spectrum.shape}") # Should be N/2 + 1

    # Create an IRFFT plan
    irfft_plan = dsp.create_irfft_plan(
        length=len(signal_in_real), # Target output length N
        precision=ods.Precision.SINGLE
    )
    signal_out_real = irfft_plan.execute(complex_spectrum)
    print(f"IRFFT Output shape: {signal_out_real.shape}") # Should be N

except RuntimeError as e:
    print(f"FFT Plan Error: {e}")
except AttributeError as e:
    print(f"API Error: Method might not exist yet ({e})")


# --- CQT Example ---
try:
    # Window function example using numpy
    def numpy_hann_window(arr):
        return np.hanning(len(arr)).astype(arr.dtype)

    # Create CQT plan using the factory method on the dsp instance
    cqt_plan = dsp.create_cqt_plan(
        sample_rate=44100.0,
        hop_length=512,
        lowest_freq=55.0,
        highest_freq=1760.0,
        bins_per_octave=12,
        window_function=numpy_hann_window,
        precision=ods.Precision.SINGLE
    )
    # Generate a test signal
    sr = 44100.0
    test_signal = np.sin(2 * np.pi * 440.0 * np.arange(int(sr)) / sr).astype(np.float32)
    cqt_result = cqt_plan.execute(test_signal) # Input is real for CQT
    print(f"CQT Output shape: {cqt_result.shape} (Bins, Frames)")
except RuntimeError as e:
    print(f"CQT Error (check backend/params): {e}")
except AttributeError as e:
    print(f"API Error: Method might not exist yet ({e})")


# --- Convolution Example (Stateless method on dsp object) ---
signal = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
kernel = np.array([0.5, 1.0, 0.5], dtype=np.float32)
# Correlate (e.g., for FIR filtering)
try:
    # Call method on the dsp instance
    correlation_result = dsp.correlate1d(signal, kernel) # Default mode is 'valid'
    print(f"Correlation result: {correlation_result}")
    # Convolve
    convolution_result = dsp.convolve1d(signal, kernel) # Default mode is 'valid'
    print(f"Convolution result: {convolution_result}")
except AttributeError as e:
    print(f"API Error: Method might not exist yet ({e})")

# --- Windowing Example (Stateless method on dsp object) ---
try:
    # Call method on the dsp instance to get coefficients
    hann_coeffs = dsp.get_hann_coeffs(len(signal), ods.Precision.SINGLE)
    print(f"Hann coeffs: {hann_coeffs}")
    # Apply manually
    windowed_signal = signal * hann_coeffs
    print(f"Hann windowed signal: {windowed_signal}")
except AttributeError as e:
    print(f"API Error: Method might not exist yet ({e})")

```

### C++

Include necessary headers (e.g., `<OmniDSP/omnidsp.h>`, `<OmniDSP/fft.h>`) and link against the built library.

```cpp
#include <OmniDSP/omnidsp.h>
#include <OmniDSP/fft.h> // Include Plan headers
#include <OmniDSP/cqt.h>
#include <vector>
#include <complex>
#include <iostream>
#include <memory> // For std::unique_ptr

int main() {
    // 1. Create an OmniDSP instance
    OmniDSP::OmniDSP dsp; // Constructor selects and initializes backend

    // Example: FFT
    std::vector<float> real_signal = { /* ... data ... */ };
    std::vector<std::complex<float>> spectrum(real_signal.size() / 2 + 1); // Pre-allocate

    try {
        // Create plan using factory method on the dsp instance
        auto rfft_plan = dsp.create_rfft_plan<float>(
            real_signal.size(),
            OmniDSP::Precision::SINGLE
        );

        // Execute the plan
        rfft_plan->execute(real_signal, spectrum); // Use -> for unique_ptr
        std::cout << "Spectrum size: " << spectrum.size() << std::endl;

        // Example: Get Hann coefficients using method on dsp instance
        auto hann_coeffs = dsp.get_hann_coeffs<float>(1024);
        std::cout << "Generated Hann coeffs size: " << hann_coeffs.size() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "OmniDSP Error: " << e.what() << std::endl;
        return 1;
    }

    // Example: CQT Plan (requires a window generator function)
    // auto my_window_gen = [](size_t len) { /* ... return std::vector<float>(len) ... */ };
    // try {
    //     // Create plan using factory method on the dsp instance
    //     auto cqt_plan = dsp.create_cqt_plan<float>(
    //         44100.0, 512, 55.0, 1760.0, 12, my_window_gen, OmniDSP::Precision::SINGLE
    //     );
    //     std::vector<std::vector<std::complex<float>>> cqt_output;
    //     cqt_plan->execute(real_signal, cqt_output);
    //     std::cout << "CQT Output bins: " << cqt_output.size() << std::endl;
    // } catch (const std::exception& e) {
    //     std::cerr << "CQT Error: " << e.what() << std::endl;
    // }


    return 0;
}
```

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines on setting up the development environment, building, running tests, and submitting pull requests. Check `TODO.md` for current tasks.

## License

_(Add License Information Here - e.g., MIT, Apache 2.0)_

# OmniDSP

OmniDSP is a C++ library designed for high-performance Digital Signal Processing tasks, with Python bindings provided via pybind11. It aims to offer efficient implementations of common DSP algorithms, leveraging optimized backend libraries like Intel oneMKL (including IPP) and Apple Accelerate where available.

## Features

- **Fast Fourier Transforms (FFT):**
  - Complex-to-Complex (C2C) FFT and IFFT.
  - Real-to-Complex (RFFT) and Complex-to-Real (IRFFT).
  - Configurable normalization modes (`BACKWARD`, `ORTHO`, `FORWARD`).
  - Uses `FFTPlan` objects for efficient repeated transforms.
- **Constant-Q Transform (CQT):**
  - Efficient recursive implementation based on FFTs and downsampling.
  - Precomputed sparse kernels for reduced runtime computation.
  - Configurable parameters (sample rate, hop length, frequency range, bins per octave, window function, sparsity).
- **Convolution / Correlation:**
  - 1D linear convolution and correlation (`valid` mode currently implemented).
- **Window Functions:**
  - Application of standard windows (Hann, Hamming, Kaiser, Flattop) via `OmniDSP::Window` class.
- **Resampling:**
  - Combined FIR filtering and downsampling (currently used internally by CQT).
- **Backend System:**
  - Automatically selects optimized backends (Intel oneMKL/IPP, Apple Accelerate) at build time based on platform and availability within the Conda environment.
  - Provides a stub backend (throws runtime errors) if no optimized backend is found.
- **Python Bindings:** Easy-to-use Python API mirroring the C++ functionality.

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

- Core FFT, CQT (recursive), Convolution/Correlation (valid mode), and Windowing functionalities are implemented.
- Backend system refactored for better organization and parity (MKL, Accelerate, Stub).
- Build system uses CMake and Conda for managing dependencies.
- **Current Work:** Addressing remaining test failures (see `TODO.md`), implementing additional convolution modes, resolving IPP resampling details (filter coefficient usage, double precision), tuning CQT scaling, and setting up CI.

## Project Structure

- `include/OmniDSP/`: Public C++ headers defining the API (`fft.h`, `cqt.h`, `convolution.h`, `window.h`, `resample.h`, `omnidsp.h`).
- `src/omnidsp/`: Core C++ library implementation files defining the public API wrappers (`fft.cpp`, `cqt.cpp`, `convolution.cpp`, `window.cpp`, `resample.cpp`).
- `src/omnidsp/backend/`: Contains backend-specific implementations.
  - `backend_impl.h`: Declares the interface required by all backends.
  - `onemkl/`: Implementations using Intel oneMKL/IPP.
  - `accelerate/`: Implementations using Apple Accelerate/vDSP.
  - `stub/`: Stub implementations that throw runtime errors.
- `src/omnidsp_py/`: Python bindings source code (pybind11 wrappers).
- `tests/`: Unit tests (primarily for developers).
  - `cpp/`: C++ tests using GoogleTest.
  - `python/`: Python tests using pytest.
- `examples/`: Usage examples.
  - `cpp/`: C++ examples.
  - `notebooks/`: Python examples using Jupyter notebooks.
- `environment.yml`: Conda environment definition source (for the `omnidsp` runtime environment).
- `environment-dev.yml`: Conda environment definition source (for the `omnidsp-dev` developer environment).
- `conda-*.lock`: Explicit, platform-specific Conda lock files for reproducible installation of the `omnidsp` runtime environment.
- `.pre-commit-config.yaml`: Configuration for developer Git hooks (see `CONTRIBUTING.md`).
- `CMakeLists.txt`: Main CMake build script.
- `pyproject.toml`: Python packaging configuration (uses `scikit-build-core`).
- `TODO.md`: Current development tasks.
- `CONTRIBUTING.md`: Guidelines for contributors.

## Getting Started & Installation

Using OmniDSP (especially the Python bindings) requires **Conda** to manage dependencies (Compiler, CMake, MKL, IPP, Python libs).

1.  **Set up Conda Environment:**

    - Clone the repository:

      ```
      git clone https://github.com/your-username/OmniDSP.git # Replace with actual URL
      cd OmniDSP
      ```

    - Create and activate the environment using the provided lock files for reproducibility. Choose the file matching your platform (e.g., `linux-64`, `osx-arm64`, `win-64`):

      ```
      # Example for Linux:
      conda create --name omnidsp --file conda-linux-64.lock
      # Example for macOS (Apple Silicon):
      # conda create --name omnidsp --file conda-osx-arm64.lock
      # Example for Windows:
      # conda create --name omnidsp --file conda-win-64.lock

      conda activate omnidsp
      ```

      _(Alternatively, for the latest definitions but less reproducibility, you could use `conda env create -f environment.yml` and then `conda activate omnidsp`)_

    - **Important:** Always ensure the `omnidsp` Conda environment is activated before building or using the library.

2.  **Install OmniDSP Python Package:**

    - Once the `omnidsp` environment is activated, install the package from the project root directory:

      ```
      pip install . -v
      ```

    - This command uses `scikit-build-core` and CMake to compile the C++ library and Python bindings, installing them into your active `omnidsp` environment. The `-v` flag provides verbose output.
    - _(Note: For development, using `pip install -e . -v` enables an "editable" install. See `CONTRIBUTING.md`)_

3.  **(Optional) Build C++ Library Only:**
    - If you only need the C++ library for use in another C++ project, please refer to the developer guidelines in `CONTRIBUTING.md` for instructions using CMake directly.

## Basic Usage

### Python

```python
import numpy as np
import omnidsp_py as ods # Assuming this is the import name after installation

# --- FFT Example ---
fs = 1000
t = np.arange(fs) / fs
signal_in = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
complex_spectrum = ods.rfft(signal_in.astype(np.float32)) # Use rfft for real input
print(f"RFFT Output shape: {complex_spectrum.shape}")
signal_out = ods.irfft(complex_spectrum)
print(f"IRFFT Output shape: {signal_out.shape}")

# --- CQT Example ---
try:
    # CQT requires compatible hop length based on octaves
    cqt_plan = ods.create_cqt_plan(sample_rate=44100.0,
                                   hop_length=512, # Divisible by 16 for A1-A6 range
                                   lowest_freq=55.0, # A1
                                   highest_freq=1760.0, # A6
                                   bins_per_octave=12)
    # Generate a test signal (e.g., 1 second sine wave at 440Hz)
    sr = 44100.0
    test_signal = np.sin(2 * np.pi * 440.0 * np.arange(int(sr)) / sr).astype(np.float32)
    cqt_result = cqt_plan.execute(test_signal)
    print(f"CQT Output shape: {cqt_result.shape} (Bins, Frames)")
except RuntimeError as e:
    print(f"CQT Error (check backend/params): {e}")


# --- Convolution Example ---
signal = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
kernel = np.array([0.5, 1.0, 0.5], dtype=np.float32)
# Correlate (e.g., for FIR filtering)
correlation_result = ods.correlate1d(signal, kernel)
print(f"Correlation result: {correlation_result}")
# Convolve
convolution_result = ods.convolve1d(signal, kernel)
print(f"Convolution result: {convolution_result}")

# --- Windowing Example ---
windowed_signal = ods.Window.hann(signal)
print(f"Hann windowed signal: {windowed_signal}")
```

### C++

Include necessary headers (e.g., `<OmniDSP/omnidsp.h>`, `<OmniDSP/cqt.h>`) and link against the built library. See `examples/cpp/` for detailed usage.

```cpp
#include <OmniDSP/omnidsp.h>
#include <OmniDSP/cqt.h>
#include <vector>
#include <complex>
#include <iostream>

int main() {
    // Example: FFT
    std::vector<float> real_signal = { /* ... data ... */ };
    std::vector<std::complex<float>> spectrum;
    OmniDSP::rfft(real_signal, spectrum);
    std::cout << "Spectrum size: " << spectrum.size() << std::endl;

    // Example: CQT Plan (requires a window generator function)
    // See examples/cpp/cqt.cpp for details on the generator function
    // auto my_window_gen = [](size_t len) { /* ... return std::vector<float>(len) ... */ };
    // OmniDSP::CQTPlan<float> cqt_plan(44100.0, 512, 55.0, 1760.0, 12, my_window_gen);
    // std::vector<std::vector<std::complex<float>>> cqt_output;
    // cqt_plan.execute(real_signal, cqt_output);

    return 0;
}
```

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines on setting up the development environment, building, running tests, and submitting pull requests. Check `TODO.md` for current tasks.

## License

_(Add License Information Here - e.g., MIT, Apache 2.0)_

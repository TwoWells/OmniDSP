# OmniDSP

OmniDSP is a C++ library designed for high-performance Digital Signal Processing tasks, with Python bindings provided via pybind11. It aims to offer efficient implementations of common DSP algorithms by abstracting optimized backend libraries like Intel oneMKL (including IPP) and Apple Accelerate.

**Core Design Philosophy:** OmniDSP centralizes backend management. Users create an `OmniDSP` object using `OmniDSP::create(BackendType)`, which selects and initializes the appropriate backend (e.g., `Default::Backend`, `OneMKL::Backend`). This instance serves as the primary interface for all DSP functionalities. Users configure operations using `[Type]Params` objects and then call distinct factory methods: `create_plan(...)` for stateless operations (returning a `[Type]Plan`) or `create_processor(...)` for stateful operations (returning a `[Type]Processor`). `Processor` objects manage their internal state. Advanced configurations allow dispatching operations to different backends per category using a `Dispatcher::Backend`.

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

- **Flexible Backend Management:** The `OmniDSP` class, created via `OmniDSP::create()`, handles backend selection.
  - Supports single backend mode (e.g., `IntelIPP::Backend`, `Accelerate::Backend`, `Default::Backend`).
  - Supports `Dispatcher::Backend` mode for per-operation-category backend overrides.
- **Distinct Execution Objects:**
  - **`[Type]Plan` (Stateless):** For operations like FFT. Created via `OmniDSP::create_plan(...)`. Thread-safe for concurrent `execute` if backend is reentrant.
  - **`[Type]Processor` (Stateful):** For operations like Filters, Resampling. Created via `OmniDSP::create_processor(...)`. Manages internal state, modified by `execute`. Provides `reset()` method. **Not thread-safe** for concurrent `execute` on the same instance.
- **Unified Configuration:**
  - **`[Type]Params` Structs:** Fluent interface for user configuration (in `include/OmniDSP/params/`).
  - **`[Type]Coefs` Structs:** For providing pre-calculated coefficients/kernels.
- **Internal Representations:** Uses intermediate `[Type]Setup` and `[Type]Spec` objects internally for backend communication.
- **Core Operations:** FFT (`FFTPlan`), Filtering (`FIRFilterProcessor`, `IIRFilterProcessor`, `filtfilt`), CQT (`CQTProcessor`?), Resampling (`ResampleProcessor`), Convolution/Correlation (`ConvolutionPlan`?), Windowing.
- **Optimized Backends:** Leverages MKL/IPP, Accelerate/vDSP. Includes a portable `Default::Backend`.
- **PIMPL Idiom:** Hides implementation details, ensures ABI stability.
- **Python Bindings:** Easy-to-use Python API via pybind11.

## Dependencies

- C++23 compiler, CMake (3.21+), Python (3.8+ for bindings).
- Boost C++ Libraries.
- Optional: Intel oneMKL/IPP, Apple Accelerate framework.
- Conda for dependency management is recommended.

## Project Status

- **Major Refactoring Ongoing:** Implementing `Dispatcher::Backend`, `Plan`/`Processor` distinction, `create_plan`/`create_processor` methods, internal state management (`reset()`), `[Type]Coefs` paths, IIR filters, `filtfilt`.
- See `TODO.md` for detailed current tasks.

## Project Structure

- `include/OmniDSP/`: Public API headers.
- `include/OmniDSP/params/`: `[Type]Params` structs.
- `include/OmniDSP/types/`: Domain-specific enums (e.g., `OperationCategory`).
- `src/omnidsp/`: Core C++ implementation.
  - `interface/`: `Abstract::Backend`, `Abstract::*PlanImpl`, `Abstract::*ProcessorImpl` interfaces, public `Plan`/`Processor` wrappers.
  - `default/`: `Default::Backend`, `Default::*PlanImpl`, `Default::*ProcessorImpl`. **Base for optimized backends.**
  - `dispatcher/`: `Dispatcher::Backend` implementation.
  - `accelerate/`, `onemkl/`, `intelipp/`: Optimized backends (e.g., `Accelerate::Backend`).
  - `params/`: `Params` constructors.
  - `utils/`: `Utils::create_spec` implementations.
  - `omnidsp.cpp`: `OmniDSP` class implementation.
- `src/omnidsp_py/`: Python bindings.
- `tests/`, `examples/`, `cmake/`, `docs/`.

## Getting Started and Installation

Using OmniDSP requires **Conda** to manage dependencies, especially for the C++ toolchain and optional backend libraries like Intel oneMKL/IPP.

1.  **Clone the Repository:**

    ```bash
    git clone <repository-url>
    cd OmniDSP
    ```

2.  **Set up Conda Environment:**
    We provide environment lock files for reproducible environments. Choose the file matching your platform (e.g., `linux-64`, `osx-arm64`, `win-64`). Using lock files is recommended.

    - **Using Lock Files (Recommended):**

      ```bash
      # Example for Linux x86_64:
      conda create --name omnidsp --file conda-linux-64.lock
      ```

      Replace `conda-linux-64.lock` with the appropriate file for your system (`conda-osx-arm64.lock`, `conda-win-64.lock`, etc.).

    - **Using `environment.yml` (If lock file is unavailable/problematic):**
      This file defines the core dependencies.
      ```bash
      conda env create --name omnidsp --file environment.yml
      ```
      _Note: This might result in slightly different package versions than tested._

3.  **Activate the Conda Environment:**
    You must activate the environment before building or using OmniDSP.

    ```bash
    conda activate omnidsp
    ```

    Your terminal prompt should now indicate that the `omnidsp` environment is active.

4.  **Install OmniDSP Python Package:**
    From the root directory of the cloned repository (where `pyproject.toml` is located), run pip install. The `-v` flag provides verbose output, helpful for debugging build issues.

    ```bash
    pip install . -v
    ```

    This command triggers the CMake build process via `scikit-build-core` (configured in `pyproject.toml`), compiling the C++ library and the pybind11 Python bindings. The resulting package is then installed into your active `omnidsp` Conda environment.

5.  **(Optional) Build C++ Library Only:** Refer to `CONTRIBUTING.md` for instructions using CMake directly without the Python package installation.

Now you should be able to import and use the `omnidsp_py` module in Python scripts run within the activated `omnidsp` environment.

## Basic Usage

### Python

```python
import numpy as np
import omnidsp_py as omni

# 1. Create an OmniDSP instance
try:
    dsp = omni.OmniDSP.create(omni.BackendType.Default)
except RuntimeError as e:
    print(f"Error creating OmniDSP instance: {e}")
    exit()

# --- FFT Example (Stateless Plan) ---
fs = 1000
signal_in_real = np.random.randn(fs).astype(np.float32)

try:
    # Configure using Params
    fft_params = omni.Params::FFT(length=len(signal_in_real))
    # Create a stateless Plan
    rfft_plan_e = dsp.create_plan(fft_params) # Use create_plan
    if not rfft_plan_e: raise RuntimeError(f"RFFT plan error: {rfft_plan_e.error()}")
    rfft_plan = rfft_plan_e.value()

    # Execute the plan
    complex_spectrum = rfft_plan.execute(signal_in_real)
    print(f"RFFT Output shape: {complex_spectrum.shape}")

except Exception as e:
    print(f"FFT Error: {e}")


# --- FIR Filter Example (Stateful Processor) ---
try:
    # Configure using Params
    fir_params = omni.FIRFilterParams(
        filter_type=omni.FilterType.Lowpass, order=100,
        cutoff_freq_low=100.0, sample_rate=float(fs),
        window_setup=omni.WindowSetup(omni.WindowType.Hann, 101)
    )
    # Create a stateful Processor
    fir_processor_e = dsp.create_processor(fir_params) # Use create_processor
    if not fir_processor_e: raise RuntimeError(f"FIR processor error: {fir_processor_e.error()}")
    fir_processor = fir_processor_e.value()

    # Execute the processor (modifies internal state)
    filtered_signal_chunk1 = fir_processor.execute(signal_in_real[:fs//2])
    filtered_signal_chunk2 = fir_processor.execute(signal_in_real[fs//2:])
    print(f"Filtered signal chunk1 shape: {filtered_signal_chunk1.shape}")
    print(f"Filtered signal chunk2 shape: {filtered_signal_chunk2.shape}")

    # Reset the processor's internal state to process a new stream
    fir_processor.reset()
    print("FIR Processor reset.")
    # filtered_signal_new_stream = fir_processor.execute(another_signal)

except Exception as e:
    print(f"FIR Filter Error: {e}")

```

### C++

```cpp
#include <OmniDSP/omnidsp.hpp>
#include <OmniDSP/fft.hpp> // For FFTPlan, Params::FFT
#include <OmniDSP/filter.hpp> // For FIRFilterProcessor
#include <OmniDSP/window.hpp>
#include <OmniDSP/params/fft.hpp> // For Params::FFT
#include <OmniDSP/params/fir_filter.hpp> // For FIRFilterParams
#include <OmniDSP/utils.hpp>
#include <OmniDSP/core_types.hpp>
#include <OmniDSP/types/operation.hpp>

#include <vector>
#include <complex>
#include <iostream>
#include <map>

// Helper to check std::expected results (same as before)
template <typename T>
T check_expected(OmniExpected<T>& expected_val, const std::string& error_msg) {
    if (!expected_val) {
        throw std::runtime_error(error_msg + ": Status " + std::to_string(static_cast<int>(expected_val.error())));
    }
    return std::move(expected_val.value());
}

int main() {
    try {
        // 1. Create an OmniDSP instance
        auto dsp_e = OmniDSP::OmniDSP::create(OmniDSP::BackendType::Default);
        auto dsp = check_expected(dsp_e, "Failed to create OmniDSP (Default)");

        // --- Example: FFT (Stateless Plan) ---
        std::vector<float> real_signal(1024); // Example data
        std::vector<std::complex<float>> spectrum;

        OmniDSP::Params::FFT fft_params; // Assuming default constructor + setters or direct init
        fft_params.length = static_cast<int>(real_signal.size());
        // Or: auto fft_params = OmniDSP::Params::FFT().length(1024);

        // Create stateless Plan
        auto rfft_plan_e = dsp->create_plan(fft_params); // Use create_plan
        auto rfft_plan = check_expected(rfft_plan_e, "RFFT plan");

        auto status = rfft_plan->execute(real_signal, spectrum);
        if (status != OmniDSP::Status::Success) throw std::runtime_error("RFFT exec");
        std::cout << "Spectrum size: " << spectrum.size() << std::endl;

        // --- Example: FIR Filter (Stateful Processor) ---
        float sample_rate = 1000.0f;
        int filter_order = 100;
        std::vector<float> filtered_signal;

        OmniDSP::FIRFilterParams fir_params( // Using constructor directly here
            OmniDSP::FilterType::Lowpass, filter_order, 100.0, 0.0, sample_rate,
            OmniDSP::WindowSetup(OmniDSP::WindowType::Hann, filter_order + 1)
        );
        // Or: auto fir_params = OmniDSP::FIRFilterParams().filter_type(...).order(...)...;

        // Create stateful Processor
        auto fir_processor_e = dsp->create_processor(fir_params); // Use create_processor
        auto fir_processor = check_expected(fir_processor_e, "FIR processor");

        // Execute (modifies internal state)
        status = fir_processor->execute(real_signal, filtered_signal);
        if (status != OmniDSP::Status::Success) throw std::runtime_error("FIR exec");
        std::cout << "Filtered signal size: " << filtered_signal.size() << std::endl;

        // Reset internal state
        fir_processor->reset();
        std::cout << "FIR Processor reset." << std::endl;


    } catch (const std::exception& e) {
        std::cerr << "OmniDSP Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

## Contributing

Contributions welcome! See `CONTRIBUTING.md` and `TODO.md`.

## License

_(Add License Information Here)_

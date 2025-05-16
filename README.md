# OmniDSP

OmniDSP is a C++ library designed for high-performance Digital Signal Processing tasks, with Python bindings provided via pybind11. It aims to offer efficient implementations of common DSP algorithms by abstracting optimized backend libraries like Intel oneMKL (including IPP) and Apple Accelerate.

**Core Design Philosophy:** OmniDSP centralizes backend management. Users create an `OmniDSP` object, which selects and initializes the appropriate backend (e.g., `Default::Backend`, `OneMKL::Backend`). This instance serves as the primary interface for all DSP functionalities. Users configure operations using `Params::[Type]` objects (or provide pre-calculated `Coefs::[Type]`) and then call distinct factory methods on the `OmniDSP` instance:

- `create_plan(...)` for stateless operations (e.g., FFT, Convolution, Windowing), returning a `Plan::[Type]`.
- `create_processor(...)` for stateful operations (e.g., Filters, CQT, Resampling), returning a `Processor::[Type]`.
  `Processor` objects manage their internal state and provide a `reset()` method. Advanced configurations allow dispatching operations to different backends per category using a `Dispatcher::Backend`.

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

- **Flexible Backend Management:** The `OmniDSP` class handles backend selection and initialization.
  - Supports single backend mode (e.g., `IntelIPP::Backend`, `Accelerate::Backend`, `Default::Backend`).
  - Supports `Dispatcher::Backend` mode for per-operation-category backend overrides.
- **Distinct Execution Objects:**
  - **`Plan::[Type]` (Stateless):** For operations like FFT, Convolution, Correlation, and Windowing. Created via `OmniDSP::create_plan(...)` (or `OmniDSP::create_window_plan(...)` for windows). Thread-safe for concurrent `execute` if the backend's implementation is reentrant.
  - **`Processor::[Type]` (Stateful):** For operations like Filters (FIR, IIR), CQT, and Resampling. Created via `OmniDSP::create_processor(...)`. Manages internal state, which is modified by `execute`. Provides a `reset()` method to re-initialize state. **Not thread-safe** for concurrent `execute` on the same instance.
- **Unified Configuration:**
  - **`Params::[Type]` Structs:** Fluent interface for user configuration (in `include/OmniDSP/params/`).
  - **`Window::[SpecType]` Structs:** Lightweight specifications for creating `Plan::Window` objects (e.g., `Window::Hann`, `Window::Kaiser`).
  - **`Coefs::[Type]` Structs:** For providing pre-calculated coefficients/kernels, bypassing design steps.
- **Internal Representations:** Uses intermediate `Design::[Type]` objects (converted from `Params::[Type]` via `OmniDSP::Design::create`) internally for backend communication for design-based operations.
- **Core Operations:**
  - **Stateless (Plans):** FFT (`Plan::FFT`, `Plan::RFFT`), Convolution (`Plan::Convolution`), Correlation (`Plan::Correlation`), Windowing (`Plan::Window`).
  - **Stateful (Processors):** FIR Filtering (`Processor::FIRFilter`), IIR Filtering (`Processor::IIRFilter`), CQT (`Processor::CQT`), Resampling (`Processor::Resample`).
- **Optimized Backends:** Leverages MKL/IPP, Accelerate/vDSP. Includes a portable `Default::Backend`.
- **PIMPL Idiom:** Hides implementation details, ensures ABI stability.
- **Python Bindings:** Easy-to-use Python API via pybind11.

## Dependencies

- C++23 compiler, CMake (3.21+), Python (3.8+ for bindings).
- Boost C++ Libraries.
- Optional: Intel oneMKL/IPP, Apple Accelerate framework.
- Conda for dependency management is recommended.

## Project Status

- **Active Development:** Focus on refining the new Windowing system (`Plan::Window`), standardizing backend PIMPL naming, and comprehensive testing of the `Plan`/`Processor` architecture.
- See `TODO.md` for detailed current tasks.

## Project Structure

- `include/OmniDSP/`: Public API headers.
  - `omnidsp.hpp`: Main `OmniDSP` class.
  - `fft.hpp`, `convolution.hpp`, `correlation.hpp`: Headers for `Plan` types.
  - `fir_filter.hpp`, `iir_filter.hpp`, `cqt.hpp`, `resample.hpp`: Headers for `Processor` types.
  - `window/`: Headers related to `Plan::Window` and window specifications (`specs.hpp`).
  - `params/`: `Params::[Type]` structs for operation configuration.
  - `design/`: `Design::[Type]` structs (internal representations).
  - `coefs/`: `Coefs::[Type]` structs for pre-calculated coefficients.
  - `core_types.hpp`, `status.hpp`, `expected.hpp`: Core utilities.
  - `types/`: Domain-specific enums (e.g., `OperationCategory`).
- `src/omnidsp/`: Core C++ implementation.
  - `interface/`: `Abstract::Backend`, `Abstract::[Type]PlanImpl`, `Abstract::[Type]ProcessorImpl` interfaces. Public `Plan`/`Processor` wrappers are in the public include headers.
  - `default/`: `Default::Backend` implementation. Implementations of `OmniDSP::Default::Plan::[Type]Impl` and `OmniDSP::Default::Processor::[Type]Impl` are in subdirectories like `plan/` and `processor/`. **Base for optimized backends.**
  - `dispatcher/`: `Dispatcher::Backend` implementation.
  - `accelerate/`, `onemkl/`, `intelipp/`: Optimized backend implementations. These contain implementations like `OmniDSP::[BackendName]::Plan::[Type]Impl` and `OmniDSP::[BackendName]::Processor::[Type]Impl`.
  - `params/`: `Params` constructors and validation logic.
  - `design/`: `OmniDSP::Design::create` implementations (conversion from `Params` to `Design`).
  - `window/`: Implementation details for windowing, including internal formulas in `OmniDSP::Internal::WindowFormulas`.
  - `omnidsp.cpp`: `OmniDSP` class implementation (factory methods).
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
import omnidsp_py as omni # Assuming omnidsp_py is the Python module name

# 1. Create an OmniDSP instance (selects a backend, e.g., Default)
try:
    # For simplicity, using Default backend.
    # Other options: omni.BackendType.OneMKL, omni.BackendType.Accelerate, etc.
    dsp = omni.OmniDSP.create(omni.BackendType.Default)
except RuntimeError as e:
    print(f"Error creating OmniDSP instance: {e}")
    exit()

# --- FFT Example (Stateless Plan) ---
fs = 1000
signal_duration = 1.0
n_samples = int(fs * signal_duration)
signal_in_real = np.random.randn(n_samples).astype(np.float32)

try:
    # Configure FFT using Params::FFT
    fft_params = omni.Params.FFT(length=len(signal_in_real)) # Corrected Params access

    # Create a stateless RFFT Plan
    # For plans like FFT, Convolution, Correlation, use dsp.create_plan()
    rfft_plan_e = dsp.create_plan(fft_params)
    if not rfft_plan_e:
        raise RuntimeError(f"RFFT plan creation error: {rfft_plan_e.error()}")
    rfft_plan = rfft_plan_e.value()

    # Execute the plan
    complex_spectrum = rfft_plan.execute(signal_in_real)
    print(f"RFFT Output shape: {complex_spectrum.shape}")

except Exception as e:
    print(f"FFT Error: {e}")


# --- FIR Filter Example (Stateful Processor) ---
try:
    # Configure FIR Filter using Params::FIRFilter
    fir_params = omni.Params.FIRFilter( # Corrected Params access
        filter_type=omni.FilterType.Lowpass,
        order=100,
        cutoff_freq_low=100.0,
        sample_rate=float(fs),
        window_spec=omni.Window.Hann() # Example using new window spec
    )

    # Create a stateful FIR Filter Processor
    fir_processor_e = dsp.create_processor(fir_params)
    if not fir_processor_e:
        raise RuntimeError(f"FIR processor creation error: {fir_processor_e.error()}")
    fir_processor = fir_processor_e.value()

    # Execute the processor (modifies internal state)
    chunk_size = n_samples // 2
    filtered_signal_chunk1 = fir_processor.execute(signal_in_real[:chunk_size])
    filtered_signal_chunk2 = fir_processor.execute(signal_in_real[chunk_size:])

    filtered_signal = np.concatenate((filtered_signal_chunk1, filtered_signal_chunk2))
    print(f"Filtered signal chunk1 shape: {filtered_signal_chunk1.shape}")
    print(f"Filtered signal chunk2 shape: {filtered_signal_chunk2.shape}")
    print(f"Total filtered signal shape: {filtered_signal.shape}")

    fir_processor.reset()
    print("FIR Processor reset.")

except Exception as e:
    print(f"FIR Filter Error: {e}")

# --- Windowing Example (Stateless Plan) ---
try:
    window_length = 128
    kaiser_spec = omni.Window.Kaiser(beta=5.0)

    # Create a window plan using dsp.create_window_plan_<type>()
    # Or a generic dsp.create_window_plan() if T_Data is templated/inferred
    window_plan_e = dsp.create_window_plan_float(kaiser_spec, window_length)
    if not window_plan_e:
        raise RuntimeError(f"Window plan creation error: {window_plan_e.error()}")
    window_plan = window_plan_e.value()

    data_to_window = np.ones(window_length, dtype=np.float32)
    windowed_data = np.empty_like(data_to_window)

    window_plan.execute(data_to_window, windowed_data) # out-of-place
    # window_plan.execute_inplace(data_to_window) # for in-place
    print(f"Windowed data (Kaiser beta=5.0, length={window_length}): {windowed_data[:10]}...")

    kaiser_coeffs_e = dsp.generate_window_coefficients_float(kaiser_spec, window_length)
    if not kaiser_coeffs_e:
        raise RuntimeError(f"Window coeffs generation error: {kaiser_coeffs_e.error()}")
    kaiser_coeffs = kaiser_coeffs_e.value()
    print(f"Generated Kaiser coefficients (beta=5.0, length={window_length}): {kaiser_coeffs[:10]}...")

except Exception as e:
    print(f"Windowing Error: {e}")

```

### C++

```cpp
#include <OmniDSP/omnidsp.hpp>
#include <OmniDSP/fft.hpp>         // For Plan::FFT, Plan::RFFT
#include <OmniDSP/fir_filter.hpp>  // For Processor::FIRFilter
#include <OmniDSP/iir_filter.hpp> // For Processor::IIRFilter (if used)
#include <OmniDSP/window/specs.hpp> // For Window::Hann, Window::Kaiser etc.
#include <OmniDSP/window.hpp>       // For Plan::Window
#include <OmniDSP/params/fft.hpp>
#include <OmniDSP/params/fir_filter.hpp>
#include <OmniDSP/core_types.hpp>
#include <OmniDSP/status.hpp>
#include <OmniDSP/expected.hpp>

#include <vector>
#include <complex>
#include <iostream>
#include <numeric> // For std::iota if needed for dummy data

// Helper to check std::expected results
template <typename T_Val, typename T_Err = OmniDSP::Status>
T_Val check_expected(OmniDSP::OmniExpected<T_Val, T_Err>& expected_val, const std::string& error_msg_prefix) {
    if (!expected_val) {
        std::string error_detail = "Error code: " + std::to_string(static_cast<int>(expected_val.error()));
        throw std::runtime_error(error_msg_prefix + ": " + error_detail);
    }
    return std::move(expected_val.value());
}


int main() {
    try {
        auto dsp_e = OmniDSP::OmniDSP::create(OmniDSP::BackendType::Default);
        auto dsp = check_expected(dsp_e, "Failed to create OmniDSP (Default)");

        // --- Example: FFT (Stateless Plan) ---
        size_t num_samples_fft = 1024;
        std::vector<float> real_signal_fft(num_samples_fft);
        for(size_t i = 0; i < num_samples_fft; ++i) real_signal_fft[i] = static_cast<float>(i % 100);
        std::vector<std::complex<float>> spectrum_fft;

        OmniDSP::Params::FFT fft_params;
        fft_params.length = static_cast<int>(real_signal_fft.size());

        // For plans like FFT, Convolution, Correlation, use dsp->create_plan()
        auto rfft_plan_e = dsp->create_plan(fft_params);
        auto rfft_plan = check_expected(rfft_plan_e, "RFFT plan creation failed");

        auto status_fft = rfft_plan->execute(real_signal_fft, spectrum_fft);
        if (status_fft != OmniDSP::Status::Success) {
            throw std::runtime_error("RFFT execution failed");
        }
        std::cout << "FFT: Spectrum size: " << spectrum_fft.size() << std::endl;

        // --- Example: FIR Filter (Stateful Processor) ---
        float sample_rate_fir = 1000.0f;
        int filter_order_fir = 100;
        size_t num_samples_fir = 512;
        std::vector<float> signal_fir(num_samples_fir);
        for(size_t i = 0; i < num_samples_fir; ++i) signal_fir[i] = static_cast<float>(i % 50);
        std::vector<float> filtered_signal_fir;

        OmniDSP::Window::Hann hann_spec;
        OmniDSP::Params::FIRFilter fir_params(
            OmniDSP::FilterType::Lowpass, filter_order_fir, 100.0f, 0.0f, sample_rate_fir, hann_spec
        );

        auto fir_processor_e = dsp->create_processor(fir_params);
        auto fir_processor = check_expected(fir_processor_e, "FIR processor creation failed");

        auto status_fir = fir_processor->execute(signal_fir, filtered_signal_fir);
         if (status_fir != OmniDSP::Status::Success) {
            throw std::runtime_error("FIR execution failed");
        }
        std::cout << "FIR Filter: Filtered signal size: " << filtered_signal_fir.size() << std::endl;

        fir_processor->reset();
        std::cout << "FIR Processor reset." << std::endl;

        // --- Example: Windowing (Stateless Plan) ---
        size_t window_len = 64;
        OmniDSP::Window::Kaiser kaiser_spec(5.0);

        // Create a window plan using dsp->create_window_plan<T_Data>()
        auto window_plan_e = dsp->create_window_plan<float>(kaiser_spec, window_len);
        auto window_plan = check_expected(window_plan_e, "Window plan creation failed");

        std::vector<float> ones_signal(window_len, 1.0f);
        std::vector<float> windowed_output(window_len);

        // Execute plan (out-of-place)
        // Note: Plan::Window::execute takes raw pointers and length
        auto status_win_oop = window_plan->execute(ones_signal.data(), windowed_output.data(), window_len);
        if (status_win_oop != OmniDSP::Status::Success) {
             throw std::runtime_error("Window plan (out-of-place) execution failed");
        }
        std::cout << "Windowing (OOP): First 5 Kaiser (beta=5.0) values: ";
        for(size_t i=0; i < 5 && i < windowed_output.size(); ++i) std::cout << windowed_output[i] << " ";
        std::cout << std::endl;

        // Execute plan (in-place)
        std::vector<float> inplace_signal(window_len, 2.0f); // Different data for in-place example
        auto status_win_ip = window_plan->execute_inplace(inplace_signal.data(), window_len);
        if (status_win_ip != OmniDSP::Status::Success) {
             throw std::runtime_error("Window plan (in-place) execution failed");
        }
        std::cout << "Windowing (IP): First 5 Kaiser (beta=5.0) values: ";
        for(size_t i=0; i < 5 && i < inplace_signal.size(); ++i) std::cout << inplace_signal[i] << " ";
        std::cout << std::endl;


        auto coeffs_e = dsp->generate_window_coefficients<float>(kaiser_spec, window_len);
        auto coeffs_vec = check_expected(coeffs_e, "Window coefficients generation failed");
        std::cout << "Windowing: First 5 generated Kaiser (beta=5.0) coefficients: ";
        for(size_t i=0; i < 5 && i < coeffs_vec.size(); ++i) std::cout << coeffs_vec[i] << " ";
        std::cout << std::endl;

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

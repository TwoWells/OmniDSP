/**
 * @file bindings.cpp
 * @brief Python bindings for the OmniDSP library using pybind11.
 *
 * This file defines the Python module 'omnidsp_py' which exposes the core
 * functionality of the OmniDSP C++ library (FFTPlan, CQTPlan, convenience functions,
 * window functions, enums) to Python, enabling its use with NumPy arrays.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>          // For automatic std::vector conversion (used implicitly by numpy)
#include <pybind11/complex.h>     // For automatic std::complex conversion
#include <pybind11/numpy.h>       // For NumPy array integration
#include <pybind11/functional.h>  // For std::function (needed for CQTPlan window)
#include <stdexcept>              // For throwing exceptions

// Include the OmniDSP headers to be bound
#include <OmniDSP/omnidsp.h>
#include <OmniDSP/cqt.h>
#include <OmniDSP/windows.h>

namespace py = pybind11;

// Define the Python module 'omnidsp_py'
// The first argument to PYBIND11_MODULE must match the target name in CMake (omnidsp_py)
PYBIND11_MODULE(omnidsp_py, m) {
    m.doc() = R"pbdoc(
        OmniDSP Python Bindings
        -----------------------
        .. currentmodule:: omnidsp_py
        .. autosummary::
           :toctree: _generate
           Direction
           Precision
           Domain
           NormMode
           FFTPlanFloat
           FFTPlanDouble
           CQTPlanFloat
           CQTPlanDouble
           Window
           fft_float
           fft_double
           ifft_float
           ifft_double
           rfft_float
           rfft_double
           irfft_float
           irfft_double
    )pbdoc";

    // --- Bind Enums ---

    py::enum_<OmniDSP::Direction>(m, "Direction", "Specifies the direction of the Fourier Transform.")
        .value("FORWARD", OmniDSP::Direction::FORWARD, "Forward Transform (e.g., time to frequency).")
        .value("INVERSE", OmniDSP::Direction::INVERSE, "Inverse Transform (e.g., frequency to time).")
        .export_values(); // Allows access like omnidsp_py.Direction.FORWARD

    py::enum_<OmniDSP::Precision>(m, "Precision", "Specifies the floating-point precision for calculations.")
        .value("SINGLE", OmniDSP::Precision::SINGLE, "Use float (32-bit) precision.")
        .value("DOUBLE", OmniDSP::Precision::DOUBLE, "Use double (64-bit) precision.")
        .export_values();

    py::enum_<OmniDSP::Domain>(m, "Domain", "Specifies the domain of the input/output signals.")
        .value("COMPLEX", OmniDSP::Domain::COMPLEX, "Complex-to-Complex (C2C) transform.")
        .value("REAL", OmniDSP::Domain::REAL, "Real-valued transform (R2C/C2R).")
        .export_values();

    py::enum_<OmniDSP::NormMode>(m, "NormMode", R"pbdoc(
        Specifies the normalization/scaling mode applied to the transforms.

        Controls the placement and value of scaling factors (like 1/N or 1/sqrt(N)).
        Ensures that IFFT(FFT(x)) = x regardless of the mode chosen.
    )pbdoc")
        .value("BACKWARD", OmniDSP::NormMode::BACKWARD, R"pbdoc(
            Forward transform is unscaled, Inverse transform is scaled by 1/N.
            Most common convention in signal processing. (Default mode).
        )pbdoc")
        .value("ORTHO", OmniDSP::NormMode::ORTHO, R"pbdoc(
            Forward and Inverse transforms are both scaled by 1/sqrt(N).
            Makes the transform unitary (preserves L2 norm/energy).
        )pbdoc")
        .value("FORWARD", OmniDSP::NormMode::FORWARD, R"pbdoc(
            Forward transform is scaled by 1/N, Inverse transform is unscaled. Less common.
        )pbdoc")
        .export_values();

    // --- Bind FFTPlan Class ---
    // Create type aliases for brevity
    using FFTPlanFloat = OmniDSP::FFTPlan<float>;
    using FFTPlanDouble = OmniDSP::FFTPlan<double>;

    py::class_<FFTPlanFloat>(m, "FFTPlanFloat", R"pbdoc(
        Manages a pre-calculated plan for efficient FFT execution (float precision).

        Creating an FFTPlan object involves setup overhead specific to the chosen backend
        (oneMKL or Accelerate) and the transform parameters. Reusing an existing plan
        for multiple transforms of the same type is generally much faster.

        This class is move-only in C++, but bindings behave like regular Python objects.
    )pbdoc")
        .def(py::init<size_t, OmniDSP::Precision, OmniDSP::Direction, OmniDSP::Domain, OmniDSP::NormMode>(),
             R"pbdoc(
                 Constructs and initializes an FFT plan.

                 Args:
                     length (int): The size 'N' of the transform.
                         - For Domain.COMPLEX: Number of complex points.
                         - For Domain.REAL: Number of real points (complex size will be N/2 + 1).
                         - Note: Accelerate backend requires N to be power-of-2 for REAL domain.
                     precision (Precision): Must be Precision.SINGLE for FFTPlanFloat.
                     direction (Direction): Transform direction (Direction.FORWARD or Direction.INVERSE).
                     domain (Domain): Transform domain (Domain.COMPLEX or Domain.REAL).
                     norm (NormMode, optional): Normalization mode. Defaults to NormMode.BACKWARD.
             )pbdoc",
             py::arg("length"), py::arg("precision"), py::arg("direction"), py::arg("domain"),
             py::arg("norm") = OmniDSP::NormMode::BACKWARD)
        // Note: Binding low-level pointer execute methods is complex for Python users.
        // Recommend using the bound convenience functions (fft_float, rfft_float, etc.) instead.
        .def("getLength", &FFTPlanFloat::getLength, "Gets the length 'N' associated with the plan.")
        .def("getComplexLength", &FFTPlanFloat::getComplexLength, R"pbdoc(
            Gets the length (Nc = N/2 + 1) of the complex spectrum for Domain.REAL plans.
            Returns N for Domain.COMPLEX plans.
            )pbdoc")
        .def("getDirection", &FFTPlanFloat::getDirection, "Gets the transform direction configured for this plan.")
        .def("getPrecision", &FFTPlanFloat::getPrecision, "Gets the floating-point precision configured for this plan.")
        .def("getDomain", &FFTPlanFloat::getDomain, "Gets the transform domain configured for this plan.")
        .def("getNormMode", &FFTPlanFloat::getNormMode, "Gets the normalization mode configured for this plan.");

    py::class_<FFTPlanDouble>(m, "FFTPlanDouble", R"pbdoc(
        Manages a pre-calculated plan for efficient FFT execution (double precision).

        See FFTPlanFloat for more details.
    )pbdoc")
        .def(py::init<size_t, OmniDSP::Precision, OmniDSP::Direction, OmniDSP::Domain, OmniDSP::NormMode>(),
             R"pbdoc(
                 Constructs and initializes an FFT plan.

                 Args:
                     length (int): The size 'N' of the transform.
                     precision (Precision): Must be Precision.DOUBLE for FFTPlanDouble.
                     direction (Direction): Transform direction.
                     domain (Domain): Transform domain.
                     norm (NormMode, optional): Normalization mode. Defaults to NormMode.BACKWARD.
             )pbdoc",
             py::arg("length"), py::arg("precision"), py::arg("direction"), py::arg("domain"),
             py::arg("norm") = OmniDSP::NormMode::BACKWARD)
        .def("getLength", &FFTPlanDouble::getLength, "Gets the length 'N' associated with the plan.")
        .def("getComplexLength", &FFTPlanDouble::getComplexLength, R"pbdoc(
            Gets the length (Nc = N/2 + 1) of the complex spectrum for Domain.REAL plans.
            Returns N for Domain.COMPLEX plans.
            )pbdoc")
        .def("getDirection", &FFTPlanDouble::getDirection, "Gets the transform direction configured for this plan.")
        .def("getPrecision", &FFTPlanDouble::getPrecision, "Gets the floating-point precision configured for this plan.")
        .def("getDomain", &FFTPlanDouble::getDomain, "Gets the transform domain configured for this plan.")
        .def("getNormMode", &FFTPlanDouble::getNormMode, "Gets the normalization mode configured for this plan.");

    // --- Bind CQTPlan Class ---
    using CQTPlanFloat = OmniDSP::CQTPlan<float>;
    using CQTPlanDouble = OmniDSP::CQTPlan<double>;

    // Define a type alias for the window function expected by CQTPlan
    using WindowFuncFloat = std::function<std::vector<float>(const std::vector<float>&)>;
    using WindowFuncDouble = std::function<std::vector<double>(const std::vector<double>&)>;

    py::class_<CQTPlanFloat>(m, "CQTPlanFloat", R"pbdoc(
        Manages a pre-calculated plan for efficient Constant Q Transform (CQT) execution (float precision).

        Uses a frequency-domain approach with precomputed kernels based on a single long FFT.
    )pbdoc")
        .def(py::init<double, double, double, int, WindowFuncFloat>(),
             R"pbdoc(
                 Constructs and initializes a CQT plan.

                 Args:
                     sample_rate (float): Sample rate of the input signal in Hz.
                     lowest_freq (float): Lowest frequency of interest (Hz, must be > 0).
                     highest_freq (float): Highest frequency of interest (Hz, must be <= sample_rate / 2).
                     bins_per_octave (int): Number of frequency bins per octave (must be > 0).
                     window_function (callable): A Python function that accepts a NumPy array (placeholder)
                         and returns a NumPy array containing the window coefficients of the appropriate length
                         as determined internally by the CQT algorithm for the current bin.
                         Note: The C++ plan calls this function internally; the placeholder argument passed
                         from C++ might just be a size-1 vector. The Python function must generate the
                         correct window based on the length required by the CQT bin being processed.
                         Alternatively, modify C++ to accept window type enums.
             )pbdoc",
             py::arg("sample_rate"), py::arg("lowest_freq"), py::arg("highest_freq"),
             py::arg("bins_per_octave"), py::arg("window_function"))
        // Bind execute using a lambda wrapper for NumPy array handling
        .def("execute", [](const CQTPlanFloat &plan, py::array_t<float, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<std::complex<float>> {
                 if (np_input.ndim() != 1) {
                     throw std::runtime_error("Input must be a 1D NumPy array.");
                 }
                 // Convert NumPy input to std::vector<float> (copies data)
                 std::vector<float> input_vec(np_input.data(), np_input.data() + np_input.size());

                 // Prepare output vector
                 std::vector<std::complex<float>> output_vec;

                 // Call the C++ execute method
                 plan.execute(input_vec, output_vec);

                 // Convert output std::vector<complex> to NumPy array (copies data) and return
                 // Create a results buffer owned by pybind11/python which holds the results from the C++ execute call
                 py::array_t<std::complex<float>> result = py::array_t<std::complex<float>>(output_vec.size());
                 // Get mutable buffer access to copy data into the Python array.
                 py::buffer_info buf = result.request();
                 std::complex<float>* ptr = static_cast<std::complex<float>*>(buf.ptr);
                 std::memcpy(ptr, output_vec.data(), output_vec.size() * sizeof(std::complex<float>));
                 return result;
             }, py::arg("input"), R"pbdoc(
                 Executes the CQT transform.

                 Args:
                     input (numpy.ndarray[float32]): A 1D NumPy array containing the input signal.

                 Returns:
                     numpy.ndarray[complex64]: A 1D NumPy array containing the complex CQT coefficients.
             )pbdoc")
        .def("getNumBins", &CQTPlanFloat::getNumBins, "Gets the number of CQT frequency bins.")
        .def("getSampleRate", &CQTPlanFloat::getSampleRate, "Gets the sample rate used for this plan (Hz).")
        .def("getLowestFrequency", &CQTPlanFloat::getLowestFrequency, "Gets the lowest frequency configured for this plan (Hz).")
        .def("getHighestFrequency", &CQTPlanFloat::getHighestFrequency, "Gets the highest frequency configured for this plan (Hz).")
        .def("getBinsPerOctave", &CQTPlanFloat::getBinsPerOctave, "Gets the number of bins per octave configured for this plan.")
        .def("getFFTLength", &CQTPlanFloat::getFFTLength, "Gets the length of the internal FFT used by the plan.");

    py::class_<CQTPlanDouble>(m, "CQTPlanDouble", R"pbdoc(
        Manages a pre-calculated plan for efficient CQT execution (double precision).
        See CQTPlanFloat for more details.
        )pbdoc")
        .def(py::init<double, double, double, int, WindowFuncDouble>(),
             R"pbdoc(
                 Constructs and initializes a CQT plan.

                 Args:
                     sample_rate (float): Sample rate in Hz.
                     lowest_freq (float): Lowest frequency in Hz.
                     highest_freq (float): Highest frequency in Hz.
                     bins_per_octave (int): Number of bins per octave.
                     window_function (callable): Python function returning window coefficients (see CQTPlanFloat docs).
             )pbdoc",
             py::arg("sample_rate"), py::arg("lowest_freq"), py::arg("highest_freq"),
             py::arg("bins_per_octave"), py::arg("window_function"))
        .def("execute", [](const CQTPlanDouble &plan, py::array_t<double, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<std::complex<double>> {
                 if (np_input.ndim() != 1) {
                     throw std::runtime_error("Input must be a 1D NumPy array.");
                 }
                 std::vector<double> input_vec(np_input.data(), np_input.data() + np_input.size());
                 std::vector<std::complex<double>> output_vec;
                 plan.execute(input_vec, output_vec);
                 // Copy C++ vector result into a new Python NumPy array
                 py::array_t<std::complex<double>> result = py::array_t<std::complex<double>>(output_vec.size());
                 py::buffer_info buf = result.request();
                 std::complex<double>* ptr = static_cast<std::complex<double>*>(buf.ptr);
                 std::memcpy(ptr, output_vec.data(), output_vec.size() * sizeof(std::complex<double>));
                 return result;
             }, py::arg("input"), R"pbdoc(
                 Executes the CQT transform.

                 Args:
                     input (numpy.ndarray[float64]): A 1D NumPy array containing the input signal.

                 Returns:
                     numpy.ndarray[complex128]: A 1D NumPy array containing the complex CQT coefficients.
             )pbdoc")
        .def("getNumBins", &CQTPlanDouble::getNumBins, "Gets the number of CQT frequency bins.")
        .def("getSampleRate", &CQTPlanDouble::getSampleRate, "Gets the sample rate used for this plan (Hz).")
        .def("getLowestFrequency", &CQTPlanDouble::getLowestFrequency, "Gets the lowest frequency configured for this plan (Hz).")
        .def("getHighestFrequency", &CQTPlanDouble::getHighestFrequency, "Gets the highest frequency configured for this plan (Hz).")
        .def("getBinsPerOctave", &CQTPlanDouble::getBinsPerOctave, "Gets the number of bins per octave configured for this plan.")
        .def("getFFTLength", &CQTPlanDouble::getFFTLength, "Gets the length of the internal FFT used by the plan.");


    // --- Bind Window Functions ---
    // Bind static methods within a dummy class 'Window' in Python
    py::class_<OmniDSP::Window>(m, "Window", "Provides common window functions for signal processing.")
        // Define static methods for each window type and precision
        .def_static("hann_float", [](py::array_t<float, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<float> {
            std::vector<float> input_vec(np_input.data(), np_input.data() + np_input.size());
            std::vector<float> output_vec = OmniDSP::Window::hann(input_vec);
            // Copy result to new numpy array
            py::array_t<float> result(output_vec.size(), output_vec.data());
            return result;
        }, py::arg("input"), "Applies the Hann window (float precision).")

        .def_static("hann_double", [](py::array_t<double, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<double> {
            std::vector<double> input_vec(np_input.data(), np_input.data() + np_input.size());
            std::vector<double> output_vec = OmniDSP::Window::hann(input_vec);
            py::array_t<double> result(output_vec.size(), output_vec.data());
            return result;
        }, py::arg("input"), "Applies the Hann window (double precision).")

        .def_static("hamming_float", [](py::array_t<float, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<float> {
            std::vector<float> input_vec(np_input.data(), np_input.data() + np_input.size());
            std::vector<float> output_vec = OmniDSP::Window::hamming(input_vec);
            py::array_t<float> result(output_vec.size(), output_vec.data());
            return result;
        }, py::arg("input"), "Applies the Hamming window (float precision).")

        .def_static("hamming_double", [](py::array_t<double, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<double> {
            std::vector<double> input_vec(np_input.data(), np_input.data() + np_input.size());
            std::vector<double> output_vec = OmniDSP::Window::hamming(input_vec);
            py::array_t<double> result(output_vec.size(), output_vec.data());
            return result;
        }, py::arg("input"), "Applies the Hamming window (double precision).")

        .def_static("kaiser_float", [](py::array_t<float, py::array::c_style | py::array::forcecast> np_input, float beta) -> py::array_t<float> {
            std::vector<float> input_vec(np_input.data(), np_input.data() + np_input.size());
            std::vector<float> output_vec = OmniDSP::Window::kaiser(input_vec, beta);
            py::array_t<float> result(output_vec.size(), output_vec.data());
            return result;
        }, py::arg("input"), py::arg("beta"), "Applies the Kaiser window (float precision).")

        .def_static("kaiser_double", [](py::array_t<double, py::array::c_style | py::array::forcecast> np_input, double beta) -> py::array_t<double> {
            std::vector<double> input_vec(np_input.data(), np_input.data() + np_input.size());
            std::vector<double> output_vec = OmniDSP::Window::kaiser(input_vec, beta);
            py::array_t<double> result(output_vec.size(), output_vec.data());
            return result;
        }, py::arg("input"), py::arg("beta"), "Applies the Kaiser window (double precision).")

        .def_static("flattop_float", [](py::array_t<float, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<float> {
            std::vector<float> input_vec(np_input.data(), np_input.data() + np_input.size());
            std::vector<float> output_vec = OmniDSP::Window::flattop(input_vec);
            py::array_t<float> result(output_vec.size(), output_vec.data());
            return result;
        }, py::arg("input"), "Applies the Flat-top window (float precision).")

        .def_static("flattop_double", [](py::array_t<double, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<double> {
            std::vector<double> input_vec(np_input.data(), np_input.data() + np_input.size());
            std::vector<double> output_vec = OmniDSP::Window::flattop(input_vec);
            py::array_t<double> result(output_vec.size(), output_vec.data());
            return result;
        }, py::arg("input"), "Applies the Flat-top window (double precision).");


    // --- Bind Convenience Functions ---
    // Use lambdas to handle NumPy array input/output (data is copied)

    m.def("fft_float", [](py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<std::complex<float>> {
        if (np_input.ndim() != 1) throw std::runtime_error("Input must be a 1D array");
        std::vector<std::complex<float>> input_vec(np_input.data(), np_input.data() + np_input.size());
        std::vector<std::complex<float>> output_vec;
        OmniDSP::fft(input_vec, output_vec);
        // Copy result to new numpy array
        py::array_t<std::complex<float>> result(output_vec.size(), output_vec.data());
        return result;
    }, py::arg("input"), "Performs C2C forward FFT (float, out-of-place, NormMode.BACKWARD).");

    m.def("fft_double", [](py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<std::complex<double>> {
        if (np_input.ndim() != 1) throw std::runtime_error("Input must be a 1D array");
        std::vector<std::complex<double>> input_vec(np_input.data(), np_input.data() + np_input.size());
        std::vector<std::complex<double>> output_vec;
        OmniDSP::fft(input_vec, output_vec);
        py::array_t<std::complex<double>> result(output_vec.size(), output_vec.data());
        return result;
    }, py::arg("input"), "Performs C2C forward FFT (double, out-of-place, NormMode.BACKWARD).");

    m.def("ifft_float", [](py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<std::complex<float>> {
        if (np_input.ndim() != 1) throw std::runtime_error("Input must be a 1D array");
        std::vector<std::complex<float>> input_vec(np_input.data(), np_input.data() + np_input.size());
        std::vector<std::complex<float>> output_vec;
        OmniDSP::ifft(input_vec, output_vec);
        py::array_t<std::complex<float>> result(output_vec.size(), output_vec.data());
        return result;
    }, py::arg("input"), "Performs C2C inverse FFT (float, out-of-place, NormMode.BACKWARD).");

    m.def("ifft_double", [](py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<std::complex<double>> {
        if (np_input.ndim() != 1) throw std::runtime_error("Input must be a 1D array");
        std::vector<std::complex<double>> input_vec(np_input.data(), np_input.data() + np_input.size());
        std::vector<std::complex<double>> output_vec;
        OmniDSP::ifft(input_vec, output_vec);
        py::array_t<std::complex<double>> result(output_vec.size(), output_vec.data());
        return result;
    }, py::arg("input"), "Performs C2C inverse FFT (double, out-of-place, NormMode.BACKWARD).");

    m.def("rfft_float", [](py::array_t<float, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<std::complex<float>> {
        if (np_input.ndim() != 1) throw std::runtime_error("Input must be a 1D array");
        std::vector<float> input_vec(np_input.data(), np_input.data() + np_input.size());
        std::vector<std::complex<float>> output_vec;
        OmniDSP::rfft(input_vec, output_vec);
        py::array_t<std::complex<float>> result(output_vec.size(), output_vec.data());
        return result;
    }, py::arg("real_input"), "Performs R2C forward FFT (float, out-of-place, NormMode.BACKWARD). Returns N/2+1 complex points.");

    m.def("rfft_double", [](py::array_t<double, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<std::complex<double>> {
        if (np_input.ndim() != 1) throw std::runtime_error("Input must be a 1D array");
        std::vector<double> input_vec(np_input.data(), np_input.data() + np_input.size());
        std::vector<std::complex<double>> output_vec;
        OmniDSP::rfft(input_vec, output_vec);
        py::array_t<std::complex<double>> result(output_vec.size(), output_vec.data());
        return result;
    }, py::arg("real_input"), "Performs R2C forward FFT (double, out-of-place, NormMode.BACKWARD). Returns N/2+1 complex points.");

    m.def("irfft_float", [](py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<float> {
        if (np_input.ndim() != 1) throw std::runtime_error("Input must be a 1D array");
        std::vector<std::complex<float>> input_vec(np_input.data(), np_input.data() + np_input.size());
        std::vector<float> output_vec;
        OmniDSP::irfft(input_vec, output_vec);
        py::array_t<float> result(output_vec.size(), output_vec.data());
        return result;
    }, py::arg("complex_input"), "Performs C2R inverse FFT (float, out-of-place, NormMode.BACKWARD). Input must have Hermitian symmetry.");

    m.def("irfft_double", [](py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<double> {
        if (np_input.ndim() != 1) throw std::runtime_error("Input must be a 1D array");
        std::vector<std::complex<double>> input_vec(np_input.data(), np_input.data() + np_input.size());
        std::vector<double> output_vec;
        OmniDSP::irfft(input_vec, output_vec);
        py::array_t<double> result(output_vec.size(), output_vec.data());
        return result;
    }, py::arg("complex_input"), "Performs C2R inverse FFT (double, out-of-place, NormMode.BACKWARD). Input must have Hermitian symmetry.");

    // Note: In-place convenience functions (fft_inplace, ifft_inplace) are not bound
    //       because directly modifying NumPy array data passed from Python via pointers
    //       can be complex regarding memory ownership and potential stride issues.
    //       It's generally safer for Python users to use the out-of-place versions:
    //       e.g., `output_array = omnidsp_py.fft_double(input_array)`

#ifdef VERSION_INFO
    // Expose the project version if defined by CMake (add -DVERSION_INFO=... to CXX flags)
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

} // End PYBIND11_MODULE
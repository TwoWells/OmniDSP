/**
 * @file bindings.cpp
 * @brief Python bindings for the OmniDSP library using pybind11.
 *
 * This file defines the Python module 'omnidsp_py' which exposes the core
 * functionality of the OmniDSP C++ library (FFTPlan, CQTPlan, convenience
 * functions, window functions, enums) to Python, enabling its use with NumPy
 * arrays. Uses pybind11 function overloading to provide single function names
 * (e.g., fft, Window.hann) for different C++ template specializations
 * (float/double). Handles CQTPlan window function adaptation and 2D output
 * conversion.
 */

#include <pybind11/complex.h>     // For automatic std::complex conversion
#include <pybind11/functional.h>  // For std::function (needed for CQTPlan window)
#include <pybind11/numpy.h>       // For NumPy array integration
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // For automatic std::vector conversion

#include <complex>     // Explicitly include for std::complex
#include <cstring>     // For std::memcpy
#include <functional>  // Explicitly include for std::function
#include <stdexcept>   // For throwing exceptions
#include <vector>      // Explicitly include for std::vector

// Include the OmniDSP headers to be bound
#include <OmniDSP/cqt.h>      // For CQTPlan
#include <OmniDSP/omnidsp.h>  // For FFTPlan, Enums, etc.
#include <OmniDSP/windows.h>  // For Window functions

namespace py = pybind11;

// --- Helper Functions ---

/**
 * @brief Converts a 1D std::vector<T> to a 1D NumPy array.
 * The NumPy array takes ownership of the data (a copy is made).
 * @tparam T The data type (e.g., float, double, std::complex<float>).
 * @param vec The input std::vector.
 * @return py::array_t<T> The resulting NumPy array.
 */
template <typename T>
py::array_t<T> vector_to_numpy_1d(const std::vector<T> &vec) {
  // Create a NumPy array of the same size as the vector.
  py::array_t<T> result = py::array_t<T>(vec.size());
  // Request buffer access to the NumPy array's memory.
  py::buffer_info buf = result.request();
  // Get a raw pointer to the NumPy array's data buffer.
  T *ptr = static_cast<T *>(buf.ptr);
  // Copy the data from the std::vector to the NumPy array's buffer.
  std::memcpy(ptr, vec.data(), vec.size() * sizeof(T));
  return result;  // Return the NumPy array.
}

/**
 * @brief Converts a 2D std::vector<std::vector<T>> to a 2D NumPy array.
 * The NumPy array takes ownership of the data (a copy is made).
 * Assumes the inner vectors (frames) all have the same size.
 * @tparam T The data type (e.g., std::complex<float>, std::complex<double>).
 * @param vec The input 2D std::vector (bins x frames).
 * @return py::array_t<T> The resulting 2D NumPy array.
 */
template <typename T>
py::array_t<T> vector_vector_to_numpy_2d(
    const std::vector<std::vector<T>> &vec) {
  if (vec.empty()) {
    // Return an empty 2D array if the input vector is empty.
    return py::array_t<T>({0, 0});
  }
  size_t rows = vec.size();     // Number of bins
  size_t cols = vec[0].size();  // Number of frames (assuming consistent size)

  // Create a 2D NumPy array with the determined shape.
  py::array_t<T> result = py::array_t<T>({rows, cols});
  // Request buffer access.
  py::buffer_info buf = result.request();
  // Get a raw pointer to the NumPy array's data buffer.
  T *ptr = static_cast<T *>(buf.ptr);

  // Iterate through the rows (bins) of the input vector.
  for (size_t i = 0; i < rows; ++i) {
    // Sanity check: Ensure inner vector has the expected number of columns.
    if (vec[i].size() != cols) {
      throw std::runtime_error(
          "Cannot convert vector<vector> to NumPy array: inner vectors have "
          "inconsistent sizes.");
    }
    // Copy data from the current inner vector (vec[i]) to the corresponding row
    // in the NumPy array. The destination pointer (ptr + i * cols) calculates
    // the start of the i-th row in the NumPy buffer.
    std::memcpy(ptr + i * cols, vec[i].data(), cols * sizeof(T));
  }
  return result;  // Return the 2D NumPy array.
}

// Define the Python module 'omnidsp_py'
// The first argument to PYBIND11_MODULE must match the target name in CMake
// (omnidsp_py)
PYBIND11_MODULE(omnidsp_py, m) {
  m.doc() = R"pbdoc(
         OmniDSP Python Bindings (with Overloaded Functions)
         ---------------------------------------------------
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
            fft
            ifft
            rfft
            irfft
     )pbdoc";

  // --- Bind Enums ---
  // Expose C++ enums to Python. export_values() makes enum members accessible
  // directly.
  py::enum_<OmniDSP::Direction>(
      m, "Direction", "Specifies the direction of the Fourier Transform.")
      .value("Forward", OmniDSP::Direction::Forward,
             "Forward Transform (e.g., time to frequency).")
      .value("Inverse", OmniDSP::Direction::Inverse,
             "Inverse Transform (e.g., frequency to time).")
      .export_values();

  py::enum_<OmniDSP::Precision>(
      m, "Precision",
      "Specifies the floating-point precision for calculations.")
      .value("Single", OmniDSP::Precision::Single,
             "Use float (32-bit) precision.")
      .value("Double", OmniDSP::Precision::Double,
             "Use double (64-bit) precision.")
      .export_values();

  py::enum_<OmniDSP::Domain>(
      m, "Domain", "Specifies the domain of the input/output signals.")
      .value("Complex", OmniDSP::Domain::Complex,
             "Complex-to-Complex (C2C) transform.")
      .value("Real", OmniDSP::Domain::Real, "Real-valued transform (R2C/C2R).")
      .export_values();

  py::enum_<OmniDSP::NormMode>(
      m, "NormMode",
      R"pbdoc(Specifies the normalization/scaling mode applied to the transforms.)pbdoc")
      .value("Backward", OmniDSP::NormMode::Backward,
             "Forward unscaled, Inverse scaled by 1/N. (Default).")
      .value("Ortho", OmniDSP::NormMode::Ortho,
             "Forward and Inverse scaled by 1/sqrt(N). Unitary.")
      .value("Forward", OmniDSP::NormMode::Forward,
             "Forward scaled by 1/N, Inverse unscaled.")
      .export_values();

  // --- Bind FFTPlan Class ---
  // Expose the FFTPlan class template for float and double precision
  // separately. We bind the constructor and getter methods. Execution is
  // typically done via convenience functions.
  using FFTPlanFloat = OmniDSP::FFTPlan<float>;
  using FFTPlanDouble = OmniDSP::FFTPlan<double>;

  py::class_<FFTPlanFloat>(
      m, "FFTPlanFloat",
      R"pbdoc(Manages a pre-calculated plan for efficient FFT execution (float precision).)pbdoc")
      .def(py::init<size_t, OmniDSP::Precision, OmniDSP::Direction,
                    OmniDSP::Domain, OmniDSP::NormMode>(),
           R"pbdoc(Constructs float FFT plan.)pbdoc", py::arg("length"),
           py::arg("precision"), py::arg("direction"), py::arg("domain"),
           py::arg("norm") =
               OmniDSP::NormMode::Backward)  // Default normalization mode
      // Bind getter methods
      .def("getLength", &FFTPlanFloat::getLength, "Gets the length 'N'.")
      .def("getComplexLength", &FFTPlanFloat::getComplexLength,
           "Gets complex spectrum length (N/2+1 for Real, N for Complex).")
      .def("getDirection", &FFTPlanFloat::getDirection,
           "Gets transform direction.")
      .def("getPrecision", &FFTPlanFloat::getPrecision, "Gets precision.")
      .def("getDomain", &FFTPlanFloat::getDomain, "Gets transform domain.")
      .def("getNormMode", &FFTPlanFloat::getNormMode,
           "Gets normalization mode.");

  py::class_<FFTPlanDouble>(
      m, "FFTPlanDouble",
      R"pbdoc(Manages a pre-calculated plan for efficient FFT execution (double precision).)pbdoc")
      .def(py::init<size_t, OmniDSP::Precision, OmniDSP::Direction,
                    OmniDSP::Domain, OmniDSP::NormMode>(),
           R"pbdoc(Constructs double FFT plan.)pbdoc", py::arg("length"),
           py::arg("precision"), py::arg("direction"), py::arg("domain"),
           py::arg("norm") =
               OmniDSP::NormMode::Backward)  // Default normalization mode
      // Bind getter methods
      .def("getLength", &FFTPlanDouble::getLength, "Gets the length 'N'.")
      .def("getComplexLength", &FFTPlanDouble::getComplexLength,
           "Gets complex spectrum length (N/2+1 for Real, N for Complex).")
      .def("getDirection", &FFTPlanDouble::getDirection,
           "Gets transform direction.")
      .def("getPrecision", &FFTPlanDouble::getPrecision, "Gets precision.")
      .def("getDomain", &FFTPlanDouble::getDomain, "Gets transform domain.")
      .def("getNormMode", &FFTPlanDouble::getNormMode,
           "Gets normalization mode.");

  // --- Bind CQTPlan Class ---
  // Bind the CQTPlan class template for float and double precision.
  using CQTPlanFloat = OmniDSP::CQTPlan<float>;
  using CQTPlanDouble = OmniDSP::CQTPlan<double>;
  // Alias for the C++ std::function signature expected by CQTPlan constructor.
  using CppWindowFuncFloat = std::function<std::vector<float>(size_t)>;
  using CppWindowFuncDouble = std::function<std::vector<double>(size_t)>;
  // Default values from C++ header (must match cqt.h)
  const float DEFAULT_SPARSITY_FLOAT = 1e-5f;
  const double DEFAULT_SPARSITY_DOUBLE = 1e-5;
  const int DEFAULT_FIR_ORDER =
      101;  // Matches DEFAULT_RECURSIVE_FIR_ORDER in cqt.h

  py::class_<CQTPlanFloat>(
      m, "CQTPlanFloat",
      R"pbdoc(Manages a pre-calculated plan for efficient recursive CQT execution (float precision).)pbdoc")
      // Bind constructor using a lambda function to handle Python callable
      // adaptation
      .def(
          py::init([](double sample_rate, size_t hop_length, double lowest_freq,
                      double highest_freq, int bins_per_octave,
                      py::object py_window_func, float sparsity_threshold,
                      int fir_filter_order) {  // Accept Python object and new
                                               // args
            // Ensure the provided Python object is callable
            if (!py::isinstance<py::function>(py_window_func)) {
              throw py::type_error("window_function must be a Python callable");
            }
            // Create a C++ std::function wrapper that calls the Python
            // function. This lambda captures the Python function object.
            CppWindowFuncFloat cpp_func =
                [py_func = std::move(py_window_func)](
                    size_t length) -> std::vector<float> {
              // The C++ CQTPlan calls this lambda with the required window
              // length. We need to call the *original* Python function. Since
              // the Python function likely expects a NumPy array (e.g., like
              // np.hanning), we create a dummy NumPy array of the correct size
              // and dtype to pass to it.
              py::array_t<float> dummy_arg(
                  length);  // Dummy argument with correct size
              py::object result =
                  py_func(dummy_arg);  // Call the Python function

              // Convert the Python function's result (expected to be a NumPy
              // array or compatible) back to a std::vector<float> for C++.
              try {
                py::array_t<float> result_arr =
                    py::cast<py::array_t<float>>(result);
                py::buffer_info info = result_arr.request();
                if (info.ndim != 1)
                  throw std::runtime_error(
                      "Window function must return a 1D array/list");
                // Check if the returned array has the expected length.
                if (static_cast<size_t>(info.shape[0]) != length) {
                  throw std::runtime_error(
                      "Window function returned array of wrong size (got " +
                      std::to_string(info.shape[0]) + ", expected " +
                      std::to_string(length) + ")");
                }
                float *ptr = static_cast<float *>(info.ptr);
                return std::vector<float>(
                    ptr, ptr + info.shape[0]);  // Create vector from buffer
              } catch (const py::cast_error &e) {
                throw std::runtime_error(
                    "Failed to cast window function result to NumPy float32 "
                    "array: " +
                    std::string(e.what()));
              } catch (
                  const std::exception &e) {  // Catch other potential errors
                throw std::runtime_error(
                    "Error processing window function result: " +
                    std::string(e.what()));
              }
            };  // End of C++ lambda wrapper definition

            // Construct the CQTPlanFloat instance, passing the C++ lambda
            // wrapper and other arguments.
            return std::make_unique<CQTPlanFloat>(
                sample_rate, hop_length, lowest_freq, highest_freq,
                bins_per_octave,
                cpp_func,  // Pass the C++ lambda wrapper
                sparsity_threshold, fir_filter_order);
          }),
          "Constructs float CQT plan.",  // Docstring for constructor
          // Define arguments for Python users, including new ones with defaults
          py::arg("sample_rate"), py::arg("hop_length"), py::arg("lowest_freq"),
          py::arg("highest_freq"), py::arg("bins_per_octave"),
          py::arg("window_function"),
          py::arg_v(
              "sparsity_threshold", DEFAULT_SPARSITY_FLOAT,
              "Sparsity threshold for kernels (default: 1e-5)"),  // Optional
                                                                  // arg
          py::arg_v(
              "fir_filter_order", DEFAULT_FIR_ORDER,
              "Order of FIR anti-aliasing filter (default: 101)")  // Optional
                                                                   // arg
          )
      // Bind execute method - Updated for 2D output
      .def(
          "execute",
          [](const CQTPlanFloat &plan,
             py::array_t<float, py::array::c_style | py::array::forcecast>
                 np_input) -> py::array_t<std::complex<float>> {
            // Input validation: Ensure input is a 1D NumPy array.
            if (np_input.ndim() != 1)
              throw std::runtime_error("Input must be a 1D NumPy array.");
            // Convert NumPy input array to std::vector<float>.
            std::vector<float> input_vec(np_input.data(),
                                         np_input.data() + np_input.size());
            // Declare the 2D vector that the C++ execute method will populate.
            std::vector<std::vector<std::complex<float>>> output_vec;
            // Call the C++ execute method.
            plan.execute(input_vec, output_vec);
            // Convert the resulting 2D std::vector to a 2D NumPy array.
            return vector_vector_to_numpy_2d(output_vec);
          },
          py::arg("input"),
          R"pbdoc(Executes float CQT transform. Returns a 2D NumPy array (bins x frames).)pbdoc")
      // Bind getters (including newly added ones)
      .def("getNumBins", &CQTPlanFloat::getNumBins,
           "Gets the total number of CQT frequency bins.")
      .def("getSampleRate", &CQTPlanFloat::getSampleRate,
           "Gets the initial sample rate used (Hz).")
      .def("getHopLength", &CQTPlanFloat::getHopLength,
           "Gets the initial hop length used (samples).")
      .def("getLowestFrequency", &CQTPlanFloat::getLowestFrequency,
           "Gets the lowest frequency (Hz).")
      .def("getHighestFrequency", &CQTPlanFloat::getHighestFrequency,
           "Gets the highest frequency (Hz).")
      .def("getBinsPerOctave", &CQTPlanFloat::getBinsPerOctave,
           "Gets the number of bins per octave.")
      .def("getNumOctaves", &CQTPlanFloat::getNumOctaves,
           "Gets the number of octaves processed.")
      .def("getSparsityThreshold", &CQTPlanFloat::getSparsityThreshold,
           "Gets the sparsity threshold used for kernels.")
      .def("getFirFilterOrder", &CQTPlanFloat::getFirFilterOrder,
           "Gets the FIR anti-aliasing filter order.");
  // .def("getOctaveFFTLengths", &CQTPlanFloat::getOctaveFFTLengths, "Gets FFT
  // lengths per octave.") // Example if getter was added

  py::class_<CQTPlanDouble>(
      m, "CQTPlanDouble",
      R"pbdoc(Manages a pre-calculated plan for efficient recursive CQT execution (double precision).)pbdoc")
      // Bind constructor using a lambda function - Updated for new arguments
      .def(
          py::init(
              [](double sample_rate, size_t hop_length, double lowest_freq,
                 double highest_freq, int bins_per_octave,
                 py::object py_window_func, double sparsity_threshold,
                 int fir_filter_order) {  // Accept Python object and new args
                if (!py::isinstance<py::function>(py_window_func)) {
                  throw py::type_error(
                      "window_function must be a Python callable");
                }
                // Create the C++ std::function wrapper.
                CppWindowFuncDouble cpp_func =
                    [py_func = std::move(py_window_func)](
                        size_t length) -> std::vector<double> {
                  py::array_t<double> dummy_arg(length);  // Dummy argument
                  py::object result =
                      py_func(dummy_arg);  // Call Python function
                  try {
                    // Convert result back to std::vector<double>.
                    py::array_t<double> result_arr =
                        py::cast<py::array_t<double>>(result);
                    py::buffer_info info = result_arr.request();
                    if (info.ndim != 1)
                      throw std::runtime_error(
                          "Window function must return a 1D array/list");
                    if (static_cast<size_t>(info.shape[0]) !=
                        length) {  // Check size consistency
                      throw std::runtime_error(
                          "Window function returned array of wrong size (got " +
                          std::to_string(info.shape[0]) + ", expected " +
                          std::to_string(length) + ")");
                    }
                    double *ptr = static_cast<double *>(info.ptr);
                    return std::vector<double>(ptr, ptr + info.shape[0]);
                  } catch (const py::cast_error &e) {
                    throw std::runtime_error(
                        "Failed to cast window function result to NumPy "
                        "float64 array: " +
                        std::string(e.what()));
                  } catch (const std::exception &e) {
                    throw std::runtime_error(
                        "Error processing window function result: " +
                        std::string(e.what()));
                  }
                };  // End of C++ lambda wrapper definition
                // Construct the CQTPlanDouble instance.
                return std::make_unique<CQTPlanDouble>(
                    sample_rate, hop_length, lowest_freq, highest_freq,
                    bins_per_octave,
                    cpp_func,  // Pass the C++ lambda wrapper
                    sparsity_threshold, fir_filter_order);
              }),
          "Constructs double CQT plan.",  // Docstring for constructor
          // Define arguments for Python users, including new ones with defaults
          py::arg("sample_rate"), py::arg("hop_length"), py::arg("lowest_freq"),
          py::arg("highest_freq"), py::arg("bins_per_octave"),
          py::arg("window_function"),
          py::arg_v(
              "sparsity_threshold", DEFAULT_SPARSITY_DOUBLE,
              "Sparsity threshold for kernels (default: 1e-5)"),  // Optional
                                                                  // arg
          py::arg_v(
              "fir_filter_order", DEFAULT_FIR_ORDER,
              "Order of FIR anti-aliasing filter (default: 101)")  // Optional
                                                                   // arg
          )
      // Bind execute method - Updated for 2D output
      .def(
          "execute",
          [](const CQTPlanDouble &plan,
             py::array_t<double, py::array::c_style | py::array::forcecast>
                 np_input) -> py::array_t<std::complex<double>> {
            // Input validation and conversion.
            if (np_input.ndim() != 1)
              throw std::runtime_error("Input must be a 1D NumPy array.");
            std::vector<double> input_vec(np_input.data(),
                                          np_input.data() + np_input.size());
            // Declare 2D output vector.
            std::vector<std::vector<std::complex<double>>> output_vec;
            // Call C++ execute.
            plan.execute(input_vec, output_vec);
            // Convert 2D vector to 2D NumPy array.
            return vector_vector_to_numpy_2d(output_vec);
          },
          py::arg("input"),
          R"pbdoc(Executes double CQT transform. Returns a 2D NumPy array (bins x frames).)pbdoc")
      // Bind getters (including newly added ones)
      .def("getNumBins", &CQTPlanDouble::getNumBins,
           "Gets the total number of CQT frequency bins.")
      .def("getSampleRate", &CQTPlanDouble::getSampleRate,
           "Gets the initial sample rate used (Hz).")
      .def("getHopLength", &CQTPlanDouble::getHopLength,
           "Gets the initial hop length used (samples).")
      .def("getLowestFrequency", &CQTPlanDouble::getLowestFrequency,
           "Gets the lowest frequency (Hz).")
      .def("getHighestFrequency", &CQTPlanDouble::getHighestFrequency,
           "Gets the highest frequency (Hz).")
      .def("getBinsPerOctave", &CQTPlanDouble::getBinsPerOctave,
           "Gets the number of bins per octave.")
      .def("getNumOctaves", &CQTPlanDouble::getNumOctaves,
           "Gets the number of octaves processed.")
      .def("getSparsityThreshold", &CQTPlanDouble::getSparsityThreshold,
           "Gets the sparsity threshold used for kernels.")
      .def("getFirFilterOrder", &CQTPlanDouble::getFirFilterOrder,
           "Gets the FIR anti-aliasing filter order.");
  // .def("getOctaveFFTLengths", &CQTPlanDouble::getOctaveFFTLengths, "Gets FFT
  // lengths per octave.") // Example if getter was added

  // --- Bind Window Functions ---
  // Bind the static methods of the C++ Window class.
  // Use pybind11's function overloading feature to expose a single Python
  // function (e.g., Window.hann) that calls the correct C++ template
  // specialization based on input type.
  py::class_<OmniDSP::Window>(
      m, "Window", "Provides common window functions for signal processing.")
      // --- Hann Window ---
      // Bind float version
      .def_static(
          "hann",
          [](py::array_t<float, py::array::c_style | py::array::forcecast>
                 np_input) -> py::array_t<float> {
            std::vector<float> input_vec(np_input.data(),
                                         np_input.data() + np_input.size());
            // Call the C++ static method OmniDSP::Window::hann<float>
            std::vector<float> result_vec = OmniDSP::Window::hann(input_vec);
            // Convert result back to NumPy array
            return vector_to_numpy_1d(result_vec);
          },
          py::arg("input"),
          "Applies Hann window (float32 input/output).")  // Overload 1: float
      // Bind double version
      .def_static(
          "hann",
          [](py::array_t<double, py::array::c_style | py::array::forcecast>
                 np_input) -> py::array_t<double> {
            std::vector<double> input_vec(np_input.data(),
                                          np_input.data() + np_input.size());
            // Call the C++ static method OmniDSP::Window::hann<double>
            std::vector<double> result_vec = OmniDSP::Window::hann(input_vec);
            // Convert result back to NumPy array
            return vector_to_numpy_1d(result_vec);
          },
          py::arg("input"),
          "Applies Hann window (float64 input/output).")  // Overload 2: double

      // --- Hamming Window --- (Similar binding pattern for other windows)
      .def_static(
          "hamming",
          [](py::array_t<float, py::array::c_style | py::array::forcecast>
                 np_input) -> py::array_t<float> {
            std::vector<float> input_vec(np_input.data(),
                                         np_input.data() + np_input.size());
            return vector_to_numpy_1d(OmniDSP::Window::hamming(input_vec));
          },
          py::arg("input"), "Applies Hamming window (float32).")
      .def_static(
          "hamming",
          [](py::array_t<double, py::array::c_style | py::array::forcecast>
                 np_input) -> py::array_t<double> {
            std::vector<double> input_vec(np_input.data(),
                                          np_input.data() + np_input.size());
            return vector_to_numpy_1d(OmniDSP::Window::hamming(input_vec));
          },
          py::arg("input"), "Applies Hamming window (float64).")

      // --- Kaiser Window ---
      .def_static(
          "kaiser",
          [](py::array_t<float, py::array::c_style | py::array::forcecast>
                 np_input,
             float beta) -> py::array_t<float> {
            std::vector<float> input_vec(np_input.data(),
                                         np_input.data() + np_input.size());
            // Call C++ kaiser<float>
            return vector_to_numpy_1d(OmniDSP::Window::kaiser(input_vec, beta));
          },
          py::arg("input"), py::arg("beta"), "Applies Kaiser window (float32).")
      .def_static(
          "kaiser",
          [](py::array_t<double, py::array::c_style | py::array::forcecast>
                 np_input,
             double beta) -> py::array_t<double> {
            std::vector<double> input_vec(np_input.data(),
                                          np_input.data() + np_input.size());
            // Call C++ kaiser<double>
            return vector_to_numpy_1d(OmniDSP::Window::kaiser(input_vec, beta));
          },
          py::arg("input"), py::arg("beta"), "Applies Kaiser window (float64).")

      // --- Flattop Window ---
      .def_static(
          "flattop",
          [](py::array_t<float, py::array::c_style | py::array::forcecast>
                 np_input) -> py::array_t<float> {
            std::vector<float> input_vec(np_input.data(),
                                         np_input.data() + np_input.size());
            return vector_to_numpy_1d(OmniDSP::Window::flattop(input_vec));
          },
          py::arg("input"), "Applies Flat-top window (float32).")
      .def_static(
          "flattop",
          [](py::array_t<double, py::array::c_style | py::array::forcecast>
                 np_input) -> py::array_t<double> {
            std::vector<double> input_vec(np_input.data(),
                                          np_input.data() + np_input.size());
            return vector_to_numpy_1d(OmniDSP::Window::flattop(input_vec));
          },
          py::arg("input"), "Applies Flat-top window (float64).");

  // --- Bind Convenience Functions ---
  // Bind the C++ convenience functions (fft, ifft, rfft, irfft).
  // Use pybind11 function overloading to handle float/double versions based on
  // NumPy input type.

  // --- FFT (C2C Forward) ---
  m.def(
      "fft",
      [](py::array_t<std::complex<float>,
                     py::array::c_style | py::array::forcecast>
             np_input) -> py::array_t<std::complex<float>> {
        // Input validation: ensure 1D array.
        if (np_input.ndim() != 1)
          throw std::runtime_error("Input must be a 1D array");
        // Convert NumPy array to std::vector.
        std::vector<std::complex<float>> input_vec(
            np_input.data(), np_input.data() + np_input.size());
        std::vector<std::complex<float>>
            output_vec;  // Output vector to be filled by C++ function.
        OmniDSP::fft(input_vec,
                     output_vec);  // Calls C++ OmniDSP::fft<float>
        // Convert result back to NumPy array.
        return vector_to_numpy_1d(output_vec);
      },
      py::arg("input"),
      "Performs C2C forward FFT (complex64 input/output, "
      "NormMode.Backward).");  // Overload 1: float

  m.def(
      "fft",
      [](py::array_t<std::complex<double>,
                     py::array::c_style | py::array::forcecast>
             np_input) -> py::array_t<std::complex<double>> {
        // Input validation and conversion.
        if (np_input.ndim() != 1)
          throw std::runtime_error("Input must be a 1D array");
        std::vector<std::complex<double>> input_vec(
            np_input.data(), np_input.data() + np_input.size());
        std::vector<std::complex<double>> output_vec;
        OmniDSP::fft(input_vec,
                     output_vec);  // Calls C++ OmniDSP::fft<double>
        // Convert result back to NumPy array.
        return vector_to_numpy_1d(output_vec);
      },
      py::arg("input"),
      "Performs C2C forward FFT (complex128 input/output, "
      "NormMode.Backward).");  // Overload 2: double

  // --- IFFT (C2C Inverse) --- (Similar binding pattern)
  m.def(
      "ifft",
      [](py::array_t<std::complex<float>,
                     py::array::c_style | py::array::forcecast>
             np_input) -> py::array_t<std::complex<float>> {
        if (np_input.ndim() != 1)
          throw std::runtime_error("Input must be a 1D array");
        std::vector<std::complex<float>> input_vec(
            np_input.data(), np_input.data() + np_input.size());
        std::vector<std::complex<float>> output_vec;
        OmniDSP::ifft(input_vec, output_vec);  // Calls C++ OmniDSP::ifft<float>
        return vector_to_numpy_1d(output_vec);
      },
      py::arg("input"),
      "Performs C2C inverse FFT (complex64 input/output, NormMode.Backward).");

  m.def(
      "ifft",
      [](py::array_t<std::complex<double>,
                     py::array::c_style | py::array::forcecast>
             np_input) -> py::array_t<std::complex<double>> {
        if (np_input.ndim() != 1)
          throw std::runtime_error("Input must be a 1D array");
        std::vector<std::complex<double>> input_vec(
            np_input.data(), np_input.data() + np_input.size());
        std::vector<std::complex<double>> output_vec;
        OmniDSP::ifft(input_vec,
                      output_vec);  // Calls C++ OmniDSP::ifft<double>
        return vector_to_numpy_1d(output_vec);
      },
      py::arg("input"),
      "Performs C2C inverse FFT (complex128 input/output, NormMode.Backward).");

  // --- RFFT (R2C Forward) ---
  m.def(
      "rfft",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> np_input)
          -> py::array_t<std::complex<float>> {
        if (np_input.ndim() != 1)
          throw std::runtime_error("Input must be a 1D array");
        std::vector<float> input_vec(np_input.data(),
                                     np_input.data() + np_input.size());
        std::vector<std::complex<float>> output_vec;  // Output is complex
        OmniDSP::rfft(input_vec, output_vec);  // Calls C++ OmniDSP::rfft<float>
        return vector_to_numpy_1d(output_vec);
      },
      py::arg("real_input"),
      "Performs R2C forward FFT (float32 input -> complex64 output, "
      "NormMode.Backward). Returns N/2+1 complex points.");

  m.def(
      "rfft",
      [](py::array_t<double, py::array::c_style | py::array::forcecast>
             np_input) -> py::array_t<std::complex<double>> {
        if (np_input.ndim() != 1)
          throw std::runtime_error("Input must be a 1D array");
        std::vector<double> input_vec(np_input.data(),
                                      np_input.data() + np_input.size());
        std::vector<std::complex<double>> output_vec;  // Output is complex
        OmniDSP::rfft(input_vec,
                      output_vec);  // Calls C++ OmniDSP::rfft<double>
        return vector_to_numpy_1d(output_vec);
      },
      py::arg("real_input"),
      "Performs R2C forward FFT (float64 input -> complex128 output, "
      "NormMode.Backward). Returns N/2+1 complex points.");

  // --- IRFFT (C2R Inverse) ---
  m.def(
      "irfft",
      [](py::array_t<std::complex<float>,
                     py::array::c_style | py::array::forcecast>
             np_input) -> py::array_t<float> {
        if (np_input.ndim() != 1)
          throw std::runtime_error("Input must be a 1D array");
        std::vector<std::complex<float>> input_vec(
            np_input.data(), np_input.data() + np_input.size());
        std::vector<float> output_vec;  // Output is real
        OmniDSP::irfft(input_vec,
                       output_vec);  // Calls C++ OmniDSP::irfft<float>
        return vector_to_numpy_1d(output_vec);
      },
      py::arg("complex_input"),
      "Performs C2R inverse FFT (complex64 input -> float32 output, "
      "NormMode.Backward). Input must have Hermitian symmetry.");

  m.def(
      "irfft",
      [](py::array_t<std::complex<double>,
                     py::array::c_style | py::array::forcecast>
             np_input) -> py::array_t<double> {
        if (np_input.ndim() != 1)
          throw std::runtime_error("Input must be a 1D array");
        std::vector<std::complex<double>> input_vec(
            np_input.data(), np_input.data() + np_input.size());
        std::vector<double> output_vec;  // Output is real
        OmniDSP::irfft(input_vec,
                       output_vec);  // Calls C++ OmniDSP::irfft<double>
        return vector_to_numpy_1d(output_vec);
      },
      py::arg("complex_input"),
      "Performs C2R inverse FFT (complex128 input -> float64 output, "
      "NormMode.Backward). Input must have Hermitian symmetry.");

  // --- Version Info ---
  // Embed version information into the module if defined during compilation
  // (e.g., via CMake -DVERSION_INFO=...)
#ifdef VERSION_INFO
// Helper macro to stringify preprocessor defines
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  // Default version if not defined during build
  m.attr("__version__") = "dev";
#endif

}  // End PYBIND11_MODULE definition

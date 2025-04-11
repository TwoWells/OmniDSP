/**
 * @file bindings.cpp
 * @brief Python bindings for the OmniDSP library using pybind11.
 *
 * This file defines the Python module 'omnidsp_py' which exposes the core
 * functionality of the OmniDSP C++ library (FFTPlan, CQTPlan, convenience functions,
 * window functions, enums) to Python, enabling its use with NumPy arrays.
 * Uses pybind11 function overloading to provide single function names (e.g., fft, Window.hann)
 * for different C++ template specializations (float/double). Handles CQTPlan window
 * function adaptation.
 */

 #include <pybind11/pybind11.h>
 #include <pybind11/stl.h>          // For automatic std::vector conversion
 #include <pybind11/complex.h>     // For automatic std::complex conversion
 #include <pybind11/numpy.h>       // For NumPy array integration
 #include <pybind11/functional.h>  // For std::function (needed for CQTPlan window)
 #include <stdexcept>              // For throwing exceptions
 #include <vector>                 // Explicitly include for std::vector
 #include <complex>                // Explicitly include for std::complex
 #include <functional>             // Explicitly include for std::function
 #include <cstring>                // For std::memcpy
 
 // Include the OmniDSP headers to be bound
 #include <OmniDSP/omnidsp.h>
 #include <OmniDSP/cqt.h>
 #include <OmniDSP/windows.h>
 
 namespace py = pybind11;
 
 // Helper function to copy std::vector<T> to py::array_t<T>
 // Ensures the Python array owns the data.
 template <typename T>
 py::array_t<T> vector_to_numpy(const std::vector<T>& vec) {
     // Create numpy array which Python will own
     py::array_t<T> result = py::array_t<T>(vec.size());
     // Get buffer access to write directly into the numpy array
     py::buffer_info buf = result.request();
     T* ptr = static_cast<T*>(buf.ptr);
     // Copy data from vector to numpy array
     std::memcpy(ptr, vec.data(), vec.size() * sizeof(T));
     return result;
 }
 
 
 // Define the Python module 'omnidsp_py'
 // The first argument to PYBIND11_MODULE must match the target name in CMake (omnidsp_py)
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
     py::enum_<OmniDSP::Direction>(m, "Direction", "Specifies the direction of the Fourier Transform.")
         .value("FORWARD", OmniDSP::Direction::FORWARD, "Forward Transform (e.g., time to frequency).")
         .value("INVERSE", OmniDSP::Direction::INVERSE, "Inverse Transform (e.g., frequency to time).")
         .export_values();
 
     py::enum_<OmniDSP::Precision>(m, "Precision", "Specifies the floating-point precision for calculations.")
         .value("SINGLE", OmniDSP::Precision::SINGLE, "Use float (32-bit) precision.")
         .value("DOUBLE", OmniDSP::Precision::DOUBLE, "Use double (64-bit) precision.")
         .export_values();
 
     py::enum_<OmniDSP::Domain>(m, "Domain", "Specifies the domain of the input/output signals.")
         .value("COMPLEX", OmniDSP::Domain::COMPLEX, "Complex-to-Complex (C2C) transform.")
         .value("REAL", OmniDSP::Domain::REAL, "Real-valued transform (R2C/C2R).")
         .export_values();
 
     py::enum_<OmniDSP::NormMode>(m, "NormMode", R"pbdoc(Specifies the normalization/scaling mode applied to the transforms.)pbdoc")
         .value("BACKWARD", OmniDSP::NormMode::BACKWARD, "Forward unscaled, Inverse scaled by 1/N. (Default).")
         .value("ORTHO", OmniDSP::NormMode::ORTHO, "Forward and Inverse scaled by 1/sqrt(N). Unitary.")
         .value("FORWARD", OmniDSP::NormMode::FORWARD, "Forward scaled by 1/N, Inverse unscaled.")
         .export_values();
 
     // --- Bind FFTPlan Class ---
     // Exposing getters, but relying on convenience functions for execution
     using FFTPlanFloat = OmniDSP::FFTPlan<float>;
     using FFTPlanDouble = OmniDSP::FFTPlan<double>;
 
     py::class_<FFTPlanFloat>(m, "FFTPlanFloat", R"pbdoc(Manages a pre-calculated plan for efficient FFT execution (float precision).)pbdoc")
         .def(py::init<size_t, OmniDSP::Precision, OmniDSP::Direction, OmniDSP::Domain, OmniDSP::NormMode>(),
              R"pbdoc(Constructs float FFT plan.)pbdoc",
              py::arg("length"), py::arg("precision"), py::arg("direction"), py::arg("domain"),
              py::arg("norm") = OmniDSP::NormMode::BACKWARD)
         .def("getLength", &FFTPlanFloat::getLength, "Gets the length 'N'.")
         .def("getComplexLength", &FFTPlanFloat::getComplexLength, "Gets complex spectrum length (N/2+1 for REAL, N for COMPLEX).")
         .def("getDirection", &FFTPlanFloat::getDirection, "Gets transform direction.")
         .def("getPrecision", &FFTPlanFloat::getPrecision, "Gets precision.")
         .def("getDomain", &FFTPlanFloat::getDomain, "Gets transform domain.")
         .def("getNormMode", &FFTPlanFloat::getNormMode, "Gets normalization mode.");
 
     py::class_<FFTPlanDouble>(m, "FFTPlanDouble", R"pbdoc(Manages a pre-calculated plan for efficient FFT execution (double precision).)pbdoc")
         .def(py::init<size_t, OmniDSP::Precision, OmniDSP::Direction, OmniDSP::Domain, OmniDSP::NormMode>(),
              R"pbdoc(Constructs double FFT plan.)pbdoc",
              py::arg("length"), py::arg("precision"), py::arg("direction"), py::arg("domain"),
              py::arg("norm") = OmniDSP::NormMode::BACKWARD)
         .def("getLength", &FFTPlanDouble::getLength, "Gets the length 'N'.")
         .def("getComplexLength", &FFTPlanDouble::getComplexLength, "Gets complex spectrum length (N/2+1 for REAL, N for COMPLEX).")
         .def("getDirection", &FFTPlanDouble::getDirection, "Gets transform direction.")
         .def("getPrecision", &FFTPlanDouble::getPrecision, "Gets precision.")
         .def("getDomain", &FFTPlanDouble::getDomain, "Gets transform domain.")
         .def("getNormMode", &FFTPlanDouble::getNormMode, "Gets normalization mode.");
 
 
     // --- Bind CQTPlan Class ---
     // Uses updated constructor binding logic
     using CQTPlanFloat = OmniDSP::CQTPlan<float>;
     using CQTPlanDouble = OmniDSP::CQTPlan<double>;
     // Alias for the NEW C++ std::function signature expected by CQTPlan
     using CppWindowFuncFloat = std::function<std::vector<float>(size_t)>;
     using CppWindowFuncDouble = std::function<std::vector<double>(size_t)>;
 
     py::class_<CQTPlanFloat>(m, "CQTPlanFloat", R"pbdoc(Manages a pre-calculated plan for efficient CQT execution (float precision).)pbdoc")
         // Updated py::init using a lambda to adapt Python callable
         .def(py::init([](double sample_rate, double lowest_freq, double highest_freq, int bins_per_octave,
                          py::object py_window_func) { // Accept Python object
                  if (!py::isinstance<py::function>(py_window_func)) {
                       throw py::type_error("window_function must be a Python callable");
                  }
                  // Create the C++ std::function wrapper that calls the Python function
                  CppWindowFuncFloat cpp_func =
                      [py_func = std::move(py_window_func)](size_t length) -> std::vector<float> {
                      // Call Python func: pass dummy numpy array of required size
                      py::array_t<float> dummy_arg(length);
                      py::object result = py_func(dummy_arg); // Call Python function
                      // Convert result back to std::vector<float>
                      try {
                          py::array_t<float> result_arr = py::cast<py::array_t<float>>(result);
                          py::buffer_info info = result_arr.request();
                          if (info.ndim != 1) throw std::runtime_error("Window function must return a 1D array/list");
                          if (static_cast<size_t>(info.shape[0]) != length) { // Check size consistency
                              throw std::runtime_error("Window function returned array of wrong size (got " + std::to_string(info.shape[0]) + ", expected " + std::to_string(length) + ")");
                          }
                          float *ptr = static_cast<float *>(info.ptr);
                          return std::vector<float>(ptr, ptr + info.shape[0]);
                      } catch (const py::cast_error& e) {
                          throw std::runtime_error("Failed to cast window function result to NumPy float32 array: " + std::string(e.what()));
                      } catch (const std::exception& e) { // Catch other potential errors
                          throw std::runtime_error("Error processing window function result: " + std::string(e.what()));
                      }
                  };
                  // Construct the CQTPlanFloat using the created std::function wrapper
                  return std::make_unique<CQTPlanFloat>(sample_rate, lowest_freq, highest_freq, bins_per_octave, cpp_func);
              }),
              "Constructs float CQT plan.", // Docstring for constructor
              py::arg("sample_rate"), py::arg("lowest_freq"), py::arg("highest_freq"),
              py::arg("bins_per_octave"), py::arg("window_function"))
         // Bind execute method
         .def("execute", [](const CQTPlanFloat &plan, py::array_t<float, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<std::complex<float>> {
                  if (np_input.ndim() != 1) throw std::runtime_error("Input must be a 1D NumPy array.");
                  std::vector<float> input_vec(np_input.data(), np_input.data() + np_input.size());
                  std::vector<std::complex<float>> output_vec;
                  plan.execute(input_vec, output_vec); // Call C++ execute
                  return vector_to_numpy(output_vec); // Convert result
              }, py::arg("input"), R"pbdoc(Executes float CQT transform.)pbdoc")
         // Bind getters
         .def("getNumBins", &CQTPlanFloat::getNumBins, "Gets the number of CQT frequency bins.")
         .def("getSampleRate", &CQTPlanFloat::getSampleRate, "Gets the sample rate used (Hz).")
         .def("getLowestFrequency", &CQTPlanFloat::getLowestFrequency, "Gets the lowest frequency (Hz).")
         .def("getHighestFrequency", &CQTPlanFloat::getHighestFrequency, "Gets the highest frequency (Hz).")
         .def("getBinsPerOctave", &CQTPlanFloat::getBinsPerOctave, "Gets the number of bins per octave.")
         .def("getFFTLength", &CQTPlanFloat::getFFTLength, "Gets the internal FFT length.");
 
     py::class_<CQTPlanDouble>(m, "CQTPlanDouble", R"pbdoc(Manages a pre-calculated plan for efficient CQT execution (double precision).)pbdoc")
          // Updated py::init using a lambda to adapt Python callable
          .def(py::init([](double sample_rate, double lowest_freq, double highest_freq, int bins_per_octave,
                          py::object py_window_func) { // Accept Python object
                  if (!py::isinstance<py::function>(py_window_func)) {
                      throw py::type_error("window_function must be a Python callable");
                  }
                  CppWindowFuncDouble cpp_func =
                       [py_func = std::move(py_window_func)](size_t length) -> std::vector<double> {
                      py::array_t<double> dummy_arg(length); // Create dummy numpy array
                      py::object result = py_func(dummy_arg); // Call Python function
                      try {
                          py::array_t<double> result_arr = py::cast<py::array_t<double>>(result);
                          py::buffer_info info = result_arr.request();
                          if (info.ndim != 1) throw std::runtime_error("Window function must return a 1D array/list");
                           if (static_cast<size_t>(info.shape[0]) != length) { // Check size consistency
                              throw std::runtime_error("Window function returned array of wrong size (got " + std::to_string(info.shape[0]) + ", expected " + std::to_string(length) + ")");
                          }
                          double *ptr = static_cast<double *>(info.ptr);
                          return std::vector<double>(ptr, ptr + info.shape[0]);
                      } catch (const py::cast_error& e) {
                         throw std::runtime_error("Failed to cast window function result to NumPy float64 array: " + std::string(e.what()));
                      } catch (const std::exception& e) {
                         throw std::runtime_error("Error processing window function result: " + std::string(e.what()));
                      }
                   };
                   return std::make_unique<CQTPlanDouble>(sample_rate, lowest_freq, highest_freq, bins_per_octave, cpp_func);
               }),
              "Constructs double CQT plan.", // Docstring for constructor
              py::arg("sample_rate"), py::arg("lowest_freq"), py::arg("highest_freq"),
              py::arg("bins_per_octave"), py::arg("window_function"))
          // Bind execute method
         .def("execute", [](const CQTPlanDouble &plan, py::array_t<double, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<std::complex<double>> {
                  if (np_input.ndim() != 1) throw std::runtime_error("Input must be a 1D NumPy array.");
                  std::vector<double> input_vec(np_input.data(), np_input.data() + np_input.size());
                  std::vector<std::complex<double>> output_vec;
                  plan.execute(input_vec, output_vec); // Call C++ execute
                  return vector_to_numpy(output_vec); // Convert result
              }, py::arg("input"), R"pbdoc(Executes double CQT transform.)pbdoc")
          // Bind getters
         .def("getNumBins", &CQTPlanDouble::getNumBins, "Gets the number of CQT frequency bins.")
         .def("getSampleRate", &CQTPlanDouble::getSampleRate, "Gets the sample rate used (Hz).")
         .def("getLowestFrequency", &CQTPlanDouble::getLowestFrequency, "Gets the lowest frequency (Hz).")
         .def("getHighestFrequency", &CQTPlanDouble::getHighestFrequency, "Gets the highest frequency (Hz).")
         .def("getBinsPerOctave", &CQTPlanDouble::getBinsPerOctave, "Gets the number of bins per octave.")
         .def("getFFTLength", &CQTPlanDouble::getFFTLength, "Gets the internal FFT length.");
 
 
     // --- Bind Window Functions --- (Using Overloading for static methods)
     py::class_<OmniDSP::Window>(m, "Window", "Provides common window functions for signal processing.")
         // --- Hann Window ---
         .def_static("hann", [](py::array_t<float, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<float> {
             std::vector<float> input_vec(np_input.data(), np_input.data() + np_input.size());
             return vector_to_numpy(OmniDSP::Window::hann(input_vec));
         }, py::arg("input"), "Applies Hann window (float32).") // Overload 1
         .def_static("hann", [](py::array_t<double, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<double> {
             std::vector<double> input_vec(np_input.data(), np_input.data() + np_input.size());
             return vector_to_numpy(OmniDSP::Window::hann(input_vec));
         }, py::arg("input"), "Applies Hann window (float64).") // Overload 2
 
         // --- Hamming Window ---
         .def_static("hamming", [](py::array_t<float, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<float> {
             std::vector<float> input_vec(np_input.data(), np_input.data() + np_input.size());
             return vector_to_numpy(OmniDSP::Window::hamming(input_vec));
         }, py::arg("input"), "Applies Hamming window (float32).")
         .def_static("hamming", [](py::array_t<double, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<double> {
             std::vector<double> input_vec(np_input.data(), np_input.data() + np_input.size());
             return vector_to_numpy(OmniDSP::Window::hamming(input_vec));
         }, py::arg("input"), "Applies Hamming window (float64).")
 
         // --- Kaiser Window ---
         .def_static("kaiser", [](py::array_t<float, py::array::c_style | py::array::forcecast> np_input, float beta) -> py::array_t<float> {
             std::vector<float> input_vec(np_input.data(), np_input.data() + np_input.size());
             return vector_to_numpy(OmniDSP::Window::kaiser(input_vec, beta));
         }, py::arg("input"), py::arg("beta"), "Applies Kaiser window (float32).")
         .def_static("kaiser", [](py::array_t<double, py::array::c_style | py::array::forcecast> np_input, double beta) -> py::array_t<double> {
             std::vector<double> input_vec(np_input.data(), np_input.data() + np_input.size());
             return vector_to_numpy(OmniDSP::Window::kaiser(input_vec, beta));
         }, py::arg("input"), py::arg("beta"), "Applies Kaiser window (float64).")
 
         // --- Flattop Window ---
         .def_static("flattop", [](py::array_t<float, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<float> {
             std::vector<float> input_vec(np_input.data(), np_input.data() + np_input.size());
             return vector_to_numpy(OmniDSP::Window::flattop(input_vec));
         }, py::arg("input"), "Applies Flat-top window (float32).")
         .def_static("flattop", [](py::array_t<double, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<double> {
             std::vector<double> input_vec(np_input.data(), np_input.data() + np_input.size());
             return vector_to_numpy(OmniDSP::Window::flattop(input_vec));
         }, py::arg("input"), "Applies Flat-top window (float64).");
 
 
     // --- Bind Convenience Functions --- (Using Overloading for module-level functions)
 
     // --- FFT (C2C Forward) ---
     m.def("fft", [](py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<std::complex<float>> {
         if (np_input.ndim() != 1) throw std::runtime_error("Input must be a 1D array");
         std::vector<std::complex<float>> input_vec(np_input.data(), np_input.data() + np_input.size());
         std::vector<std::complex<float>> output_vec;
         OmniDSP::fft(input_vec, output_vec); // Calls C++ OmniDSP::fft<float>
         return vector_to_numpy(output_vec);
     }, py::arg("input"), "Performs C2C forward FFT (complex64, NormMode.BACKWARD)."); // Overload 1
 
     m.def("fft", [](py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<std::complex<double>> {
         if (np_input.ndim() != 1) throw std::runtime_error("Input must be a 1D array");
         std::vector<std::complex<double>> input_vec(np_input.data(), np_input.data() + np_input.size());
         std::vector<std::complex<double>> output_vec;
         OmniDSP::fft(input_vec, output_vec); // Calls C++ OmniDSP::fft<double>
         return vector_to_numpy(output_vec);
     }, py::arg("input"), "Performs C2C forward FFT (complex128, NormMode.BACKWARD)."); // Overload 2
 
     // --- IFFT (C2C Inverse) ---
     m.def("ifft", [](py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<std::complex<float>> {
         if (np_input.ndim() != 1) throw std::runtime_error("Input must be a 1D array");
         std::vector<std::complex<float>> input_vec(np_input.data(), np_input.data() + np_input.size());
         std::vector<std::complex<float>> output_vec;
         OmniDSP::ifft(input_vec, output_vec); // Calls C++ OmniDSP::ifft<float>
         return vector_to_numpy(output_vec);
     }, py::arg("input"), "Performs C2C inverse FFT (complex64, NormMode.BACKWARD).");
 
     m.def("ifft", [](py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<std::complex<double>> {
         if (np_input.ndim() != 1) throw std::runtime_error("Input must be a 1D array");
         std::vector<std::complex<double>> input_vec(np_input.data(), np_input.data() + np_input.size());
         std::vector<std::complex<double>> output_vec;
         OmniDSP::ifft(input_vec, output_vec); // Calls C++ OmniDSP::ifft<double>
         return vector_to_numpy(output_vec);
     }, py::arg("input"), "Performs C2C inverse FFT (complex128, NormMode.BACKWARD).");
 
     // --- RFFT (R2C Forward) ---
     m.def("rfft", [](py::array_t<float, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<std::complex<float>> {
         if (np_input.ndim() != 1) throw std::runtime_error("Input must be a 1D array");
         std::vector<float> input_vec(np_input.data(), np_input.data() + np_input.size());
         std::vector<std::complex<float>> output_vec;
         OmniDSP::rfft(input_vec, output_vec); // Calls C++ OmniDSP::rfft<float>
         return vector_to_numpy(output_vec);
     }, py::arg("real_input"), "Performs R2C forward FFT (float32, NormMode.BACKWARD). Returns N/2+1 complex points.");
 
     m.def("rfft", [](py::array_t<double, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<std::complex<double>> {
         if (np_input.ndim() != 1) throw std::runtime_error("Input must be a 1D array");
         std::vector<double> input_vec(np_input.data(), np_input.data() + np_input.size());
         std::vector<std::complex<double>> output_vec;
         OmniDSP::rfft(input_vec, output_vec); // Calls C++ OmniDSP::rfft<double>
         return vector_to_numpy(output_vec);
     }, py::arg("real_input"), "Performs R2C forward FFT (float64, NormMode.BACKWARD). Returns N/2+1 complex points.");
 
     // --- IRFFT (C2R Inverse) ---
     m.def("irfft", [](py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<float> {
         if (np_input.ndim() != 1) throw std::runtime_error("Input must be a 1D array");
         std::vector<std::complex<float>> input_vec(np_input.data(), np_input.data() + np_input.size());
         std::vector<float> output_vec;
         OmniDSP::irfft(input_vec, output_vec); // Calls C++ OmniDSP::irfft<float>
         return vector_to_numpy(output_vec);
     }, py::arg("complex_input"), "Performs C2R inverse FFT (complex64, NormMode.BACKWARD). Input must have Hermitian symmetry.");
 
     m.def("irfft", [](py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> np_input) -> py::array_t<double> {
         if (np_input.ndim() != 1) throw std::runtime_error("Input must be a 1D array");
         std::vector<std::complex<double>> input_vec(np_input.data(), np_input.data() + np_input.size());
         std::vector<double> output_vec;
         OmniDSP::irfft(input_vec, output_vec); // Calls C++ OmniDSP::irfft<double>
         return vector_to_numpy(output_vec);
     }, py::arg("complex_input"), "Performs C2R inverse FFT (complex128, NormMode.BACKWARD). Input must have Hermitian symmetry.");
 
 
     // --- Version Info ---
 #ifdef VERSION_INFO // VERSION_INFO should be defined by CMake (e.g., -DVERSION_INFO=...)
     // Helper macro to stringify preprocessor defines
     #define STRINGIFY(x) #x
     #define MACRO_STRINGIFY(x) STRINGIFY(x)
     m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
 #else
     m.attr("__version__") = "dev";
 #endif
 
 } // End PYBIND11_MODULE
/**
 * @file bindings.cpp
 * @brief pybind11 bindings for the OmniDSP C++ library.
 * @details Exposes the OmniDSP class, Plan objects, and core functionalities to
 * Python.
 */

#include <pybind11/complex.h>     // For std::complex support
#include <pybind11/functional.h>  // For std::function wrapping if needed
#include <pybind11/numpy.h>       // For NumPy array support
#include <pybind11/pybind11.h>
#include <pybind11/smart_ptr.h>  // For std::unique_ptr support
#include <pybind11/stl.h>  // For automatic std::vector <-> list/tuple conversion

#include "OmniDSP/convolution.h"
#include "OmniDSP/core_types.h"
#include "OmniDSP/cqt.h"
#include "OmniDSP/fft.h"
#include "OmniDSP/omnidsp.h"  // Main C++ API header
#include "OmniDSP/resample.h"
#include "OmniDSP/window.h"
// #include "OmniDSP/filter.h" // Include when available

#include <span>  // Requires C++20
#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace OmniDSP;

// Define types for template instantiations
using float_r = RealT<float>;
using double_r = RealT<double>;
using float_c = ComplexT<float>;
using double_c = ComplexT<double>;

// Custom Exception Class for OmniDSP Errors
// Allows Python code to catch specific library errors
class OmniDSPPyError : public std::runtime_error {
 public:
  OmniDSPPyError(OmniStatus s, const std::string& message)
      : std::runtime_error(message), status_(s)
  {}

  OmniStatus get_status() const { return status_; }

 private:
  OmniStatus status_;
};

// Helper to check OmniExpected<T> and throw OmniDSPPyError on failure
template <typename T>
T check_expected(
    OmniExpected<T>&& result, const std::string& func_name = "OmniDSP function")
{
  if (result.has_value()) {
    return std::move(result.value());
  }
  else {
    OmniStatus s = result.error();
    std::string error_msg
        = func_name + " failed with status: " + get_status_string(s);
    throw OmniDSPPyError(s, error_msg);
  }
}

// Overload for OmniExpected<void>
void check_expected_void(
    OmniExpected<void>&& result,
    const std::string& func_name = "OmniDSP function")
{
  if (!result.has_value()) {
    OmniStatus s = result.error();
    std::string error_msg
        = func_name + " failed with status: " + get_status_string(s);
    throw OmniDSPPyError(s, error_msg);
  }
}

// Helper to check OmniStatus return values (for Plan execute methods)
void check_status(
    OmniStatus status, const std::string& func_name = "OmniDSP Plan execute")
{
  if (status != OmniStatus::Success) {
    std::string error_msg
        = func_name + " failed with status: " + get_status_string(status);
    throw OmniDSPPyError(status, error_msg);
  }
}

// Helper to convert NumPy array to std::span (const version)
// Performs dtype and contiguity checks.
template <typename T>
std::span<const T> numpy_to_span_const(const py::buffer& buf)
{
  py::buffer_info info = buf.request();  // Request buffer information

  // Check dimensions (expect 1D)
  if (info.ndim != 1) {
    throw std::runtime_error("Input array must be 1-dimensional.");
  }

  // Check data type
  if (!py::isinstance<py::array_t<T>>(buf)) {
    // More specific error message based on expected type T?
    throw std::runtime_error("Input array has incorrect data type.");
  }

  // Check contiguity (optional but recommended for direct span usage)
  // if (!(info.strides[0] == sizeof(T))) { // Check for C-contiguity
  //     throw std::runtime_error("Input array must be C-contiguous.");
  // }

  return std::span<const T>(static_cast<const T*>(info.ptr), info.shape[0]);
}

// Helper to convert NumPy array to std::span (non-const version)
// Performs dtype and contiguity checks.
template <typename T>
std::span<T> numpy_to_span_writable(py::buffer buf)
{
  py::buffer_info info = buf.request(true);  // Request writable buffer

  if (info.ndim != 1) {
    throw std::runtime_error("Output array must be 1-dimensional.");
  }
  if (!py::isinstance<py::array_t<T>>(buf)) {
    throw std::runtime_error("Output array has incorrect data type.");
  }
  // if (!(info.strides[0] == sizeof(T))) {
  //     throw std::runtime_error("Output array must be C-contiguous.");
  // }

  return std::span<T>(static_cast<T*>(info.ptr), info.shape[0]);
}

PYBIND11_MODULE(_omnidsp_cpp, m)
{  // Matches the target name in CMakeLists.txt
  m.doc() = "Python bindings for the OmniDSP C++ library";

  // --- Custom Exception ---
  // Register the custom exception class with pybind11
  static py::exception<OmniDSPPyError> omni_dsp_py_error(
      m, "OmniDSPError", PyExc_RuntimeError);
  py::register_exception_translator(
      [](std::exception_ptr p)
      {
        try {
          if (p) std::rethrow_exception(p);
        }
        catch (const OmniDSPPyError& e) {
          // Set the Python error object with the message from the C++ exception
          // Include the status code if desired
          std::string msg = std::string(e.what()) + " (OmniStatus Code: "
                            + std::to_string(static_cast<int>(e.get_status()))
                            + ")";
          PyErr_SetString(omni_dsp_py_error.ptr(), msg.c_str());
        }
        // Add catch blocks for other C++ exceptions if needed
      });

  // --- Enums ---
  py::enum_<BackendType>(m, "BackendType")
      .value("Default", BackendType::Default)
      .value("Accelerate", BackendType::Accelerate)
      .value("OneMKL", BackendType::OneMKL)
      .export_values();

  py::enum_<OmniStatus>(m, "OmniStatus")
      .value("Success", OmniStatus::Success)
      .value("Failure", OmniStatus::Failure)
      .value("InvalidArgument", OmniStatus::InvalidArgument)
      .value("InvalidOperation", OmniStatus::InvalidOperation)
      .value("AllocationError", OmniStatus::AllocationError)
      .value("BackendError", OmniStatus::BackendError)
      .value("UnsupportedFeature", OmniStatus::UnsupportedFeature)
      .value("SizeMismatch", OmniStatus::SizeMismatch)
      .value("OutOfBounds", OmniStatus::OutOfBounds)
      .export_values();

  py::enum_<DataType>(m, "DataType")
      .value("Real", DataType::Real)
      .value("Complex", DataType::Complex)
      .export_values();

  py::enum_<Domain>(m, "Domain")
      .value("Time", Domain::Time)
      .value("Frequency", Domain::Frequency)
      .value("Quefrency", Domain::Quefrency)
      .export_values();

  py::enum_<Window>(m, "WindowType")  // Use WindowType in Python for clarity?
      .value("Bartlett", Window::Bartlett)
      .value("Blackman", Window::Blackman)
      .value("Flattop", Window::Flattop)
      .value("Gaussian", Window::Gaussian)
      .value("Hamming", Window::Hamming)
      .value("Hann", Window::Hann)
      .value("Kaiser", Window::Kaiser)
      .value("Rectangular", Window::Rectangular)
      .value("Triangular", Window::Triangular)
      .export_values();

  py::enum_<ConvolutionMode>(m, "ConvolutionMode")
      .value("Full", ConvolutionMode::Full)
      .value("Same", ConvolutionMode::Same)
      .value("Valid", ConvolutionMode::Valid)
      .export_values();

  // --- WindowSetup ---
  // Need to bind template specializations
  py::class_<WindowSetup<float>>(m, "WindowSpecFloat")
      .def(py::init<>())  // Default (Hann)
      .def(py::init<Window>(), py::arg("type"))
      .def(py::init<Window, float>(), py::arg("type"), py::arg("param"))
      .def("get_type", &WindowSetup<float>::get_type)
      .def("get_beta", &WindowSetup<float>::get_beta)
      .def("get_stddev", &WindowSetup<float>::get_stddev)
      // Add __repr__ for better printing?
      ;

  py::class_<WindowSetup<double>>(m, "WindowSpecDouble")
      .def(py::init<>())  // Default (Hann)
      .def(py::init<Window>(), py::arg("type"))
      .def(py::init<Window, double>(), py::arg("type"), py::arg("param"))
      .def("get_type", &WindowSetup<double>::get_type)
      .def("get_beta", &WindowSetup<double>::get_beta)
      .def("get_stddev", &WindowSetup<double>::get_stddev);

  // --- Plan Classes (Bind Interfaces) ---
  // FFTPlan (Complex)
  py::class_<FFTPlan<float_c>, std::unique_ptr<FFTPlan<float_c>>>(
      m, "FFTPlanFloatComplex")
      // No constructor exposed
      .def(
          "fft",
          [](const FFTPlan<float_c>& self,
             py::array_t<float_c, py::array::c_style | py::array::forcecast>
                 input,
             py::array_t<float_c, py::array::c_style | py::array::forcecast>
                 output)
          {
            auto input_span = numpy_to_span_const<float_c>(input);
            auto output_span = numpy_to_span_writable<float_c>(output);
            check_status(self.fft(input_span, output_span), "FFTPlan.fft");
          },
          py::arg("input"),
          py::arg("output"),
          "Execute forward complex FFT.")
      .def(
          "ifft",
          [](const FFTPlan<float_c>& self,
             py::array_t<float_c, py::array::c_style | py::array::forcecast>
                 input,
             py::array_t<float_c, py::array::c_style | py::array::forcecast>
                 output)
          {
            auto input_span = numpy_to_span_const<float_c>(input);
            auto output_span = numpy_to_span_writable<float_c>(output);
            check_status(self.ifft(input_span, output_span), "FFTPlan.ifft");
            // Note: No automatic 1/N scaling applied here, matches C++ Plan
          },
          py::arg("input"),
          py::arg("output"),
          "Execute inverse complex FFT (unscaled).")
      .def("get_length", &FFTPlan<float_c>::get_length);

  py::class_<FFTPlan<double_c>, std::unique_ptr<FFTPlan<double_c>>>(
      m, "FFTPlanDoubleComplex")
      .def(
          "fft",
          [](const FFTPlan<double_c>& self,
             py::array_t<double_c, py::array::c_style | py::array::forcecast>
                 input,
             py::array_t<double_c, py::array::c_style | py::array::forcecast>
                 output)
          {
            check_status(
                self.fft(
                    numpy_to_span_const<double_c>(input),
                    numpy_to_span_writable<double_c>(output)),
                "FFTPlan.fft");
          },
          py::arg("input"),
          py::arg("output"),
          "Execute forward complex FFT.")
      .def(
          "ifft",
          [](const FFTPlan<double_c>& self,
             py::array_t<double_c, py::array::c_style | py::array::forcecast>
                 input,
             py::array_t<double_c, py::array::c_style | py::array::forcecast>
                 output)
          {
            check_status(
                self.ifft(
                    numpy_to_span_const<double_c>(input),
                    numpy_to_span_writable<double_c>(output)),
                "FFTPlan.ifft");
          },
          py::arg("input"),
          py::arg("output"),
          "Execute inverse complex FFT (unscaled).")
      .def("get_length", &FFTPlan<double_c>::get_length);

  // RFFTPlan (Real)
  py::class_<RFFTPlan<float>, std::unique_ptr<RFFTPlan<float>>>(
      m, "RFFTPlanFloat")
      .def(
          "rfft",
          [](const RFFTPlan<float>& self,
             py::array_t<float_r, py::array::c_style | py::array::forcecast>
                 input,
             py::array_t<float_c, py::array::c_style | py::array::forcecast>
                 output)
          {
            check_status(
                self.rfft(
                    numpy_to_span_const<float_r>(input),
                    numpy_to_span_writable<float_c>(output)),
                "RFFTPlan.rfft");
          },
          py::arg("input"),
          py::arg("output"),
          "Execute forward real-to-complex FFT.")
      .def(
          "irfft",
          [](const RFFTPlan<float>& self,
             py::array_t<float_c, py::array::c_style | py::array::forcecast>
                 input,
             py::array_t<float_r, py::array::c_style | py::array::forcecast>
                 output)
          {
            check_status(
                self.irfft(
                    numpy_to_span_const<float_c>(input),
                    numpy_to_span_writable<float_r>(output)),
                "RFFTPlan.irfft");
            // Note: No automatic 1/N scaling applied here
          },
          py::arg("input"),
          py::arg("output"),
          "Execute inverse complex-to-real FFT (unscaled).")
      .def("get_length", &RFFTPlan<float>::get_length);

  py::class_<RFFTPlan<double>, std::unique_ptr<RFFTPlan<double>>>(
      m, "RFFTPlanDouble")
      .def(
          "rfft",
          [](const RFFTPlan<double>& self,
             py::array_t<double_r, py::array::c_style | py::array::forcecast>
                 input,
             py::array_t<double_c, py::array::c_style | py::array::forcecast>
                 output)
          {
            check_status(
                self.rfft(
                    numpy_to_span_const<double_r>(input),
                    numpy_to_span_writable<double_c>(output)),
                "RFFTPlan.rfft");
          },
          py::arg("input"),
          py::arg("output"),
          "Execute forward real-to-complex FFT.")
      .def(
          "irfft",
          [](const RFFTPlan<double>& self,
             py::array_t<double_c, py::array::c_style | py::array::forcecast>
                 input,
             py::array_t<double_r, py::array::c_style | py::array::forcecast>
                 output)
          {
            check_status(
                self.irfft(
                    numpy_to_span_const<double_c>(input),
                    numpy_to_span_writable<double_r>(output)),
                "RFFTPlan.irfft");
          },
          py::arg("input"),
          py::arg("output"),
          "Execute inverse complex-to-real FFT (unscaled).")
      .def("get_length", &RFFTPlan<double>::get_length);

  // CQTProcessor
  py::class_<CQTProcessor<float>, std::unique_ptr<CQTProcessor<float>>>(
      m, "CQTPlanFloat")
      .def(
          "execute",
          [](const CQTProcessor<float>& self,
             py::array_t<float_r, py::array::c_style | py::array::forcecast>
                 input,
             py::array_t<float_c, py::array::c_style | py::array::forcecast>
                 output)
          {
            check_status(
                self.execute(
                    numpy_to_span_const<float_r>(input),
                    numpy_to_span_writable<float_c>(output)),
                "CQTProcessor.execute");
          },
          py::arg("input"),
          py::arg("output"),
          "Execute Constant-Q Transform.")
      .def("get_num_bins", &CQTProcessor<float>::get_num_bins)
      .def(
          "get_num_output_frames",
          &CQTProcessor<float>::get_num_output_frames,
          py::arg("input_length"))
      .def("get_hop_length", &CQTProcessor<float>::get_hop_length);

  py::class_<CQTProcessor<double>, std::unique_ptr<CQTProcessor<double>>>(
      m, "CQTPlanDouble")
      .def(
          "execute",
          [](const CQTProcessor<double>& self,
             py::array_t<double_r, py::array::c_style | py::array::forcecast>
                 input,
             py::array_t<double_c, py::array::c_style | py::array::forcecast>
                 output)
          {
            check_status(
                self.execute(
                    numpy_to_span_const<double_r>(input),
                    numpy_to_span_writable<double_c>(output)),
                "CQTProcessor.execute");
          },
          py::arg("input"),
          py::arg("output"),
          "Execute Constant-Q Transform.")
      .def("get_num_bins", &CQTProcessor<double>::get_num_bins)
      .def(
          "get_num_output_frames",
          &CQTProcessor<double>::get_num_output_frames,
          py::arg("input_length"))
      .def("get_hop_length", &CQTProcessor<double>::get_hop_length);

  // ConvolutionPlan
  py::class_<ConvolutionPlan<float>, std::unique_ptr<ConvolutionPlan<float>>>(
      m, "ConvolutionPlanFloat")
      .def(
          "execute",
          [](const ConvolutionPlan<float>& self,
             py::array_t<float, py::array::c_style | py::array::forcecast>
                 input,
             py::array_t<float, py::array::c_style | py::array::forcecast>
                 output)
          {
            check_status(
                self.execute(
                    numpy_to_span_const<float>(input),
                    numpy_to_span_writable<float>(output)),
                "ConvolutionPlan.execute");
          },
          py::arg("input"),
          py::arg("output"),
          "Execute pre-planned convolution.")
      .def("get_kernel_length", &ConvolutionPlan<float>::get_kernel_length)
      .def("get_mode", &ConvolutionPlan<float>::get_mode)
      .def(
          "get_output_length",
          &ConvolutionPlan<float>::get_output_length,
          py::arg("input_length"));
  // Add double, complex float, complex double specializations...

  // CorrelationPlan
  py::class_<CorrelationPlan<float>, std::unique_ptr<CorrelationPlan<float>>>(
      m, "CorrelationPlanFloat")
      .def(
          "execute",
          [](const CorrelationPlan<float>& self,
             py::array_t<float, py::array::c_style | py::array::forcecast>
                 input,
             py::array_t<float, py::array::c_style | py::array::forcecast>
                 output)
          {
            check_status(
                self.execute(
                    numpy_to_span_const<float>(input),
                    numpy_to_span_writable<float>(output)),
                "CorrelationPlan.execute");
          },
          py::arg("input"),
          py::arg("output"),
          "Execute pre-planned correlation.")
      .def("get_template_length", &CorrelationPlan<float>::get_template_length)
      .def("get_mode", &CorrelationPlan<float>::get_mode)
      .def(
          "get_output_length",
          &CorrelationPlan<float>::get_output_length,
          py::arg("input_length"));
  // Add double, complex float, complex double specializations...

  // ResampleProcessor
  py::class_<
      ResampleProcessor<float>,
      std::unique_ptr<ResampleProcessor<float>>>(m, "ResamplePlanFloat")
      .def(
          "execute",
          [](const ResampleProcessor<float>& self,
             py::array_t<float, py::array::c_style | py::array::forcecast>
                 input,
             py::array_t<float, py::array::c_style | py::array::forcecast>
                 output)
          {
            check_status(
                self.execute(
                    numpy_to_span_const<float>(input),
                    numpy_to_span_writable<float>(output)),
                "ResampleProcessor.execute");
          },
          py::arg("input"),
          py::arg("output"),
          "Execute pre-planned resampling.")
      .def("get_input_rate", &ResampleProcessor<float>::get_input_rate)
      .def("get_output_rate", &ResampleProcessor<float>::get_output_rate)
      .def(
          "get_output_length",
          &ResampleProcessor<float>::get_output_length,
          py::arg("input_length"));
  // Add double specialization...

  // --- OmniDSP Class ---
  py::class_<OmniDSP>(m, "OmniDSP")
      // Do not expose constructor directly
      .def_static(
          "create",
          [](BackendType backend = BackendType::Default)
          {
            // Use helper to handle expected and potential exception
            return check_expected(OmniDSP::create(backend), "OmniDSP.create");
          },
          py::arg("backend") = BackendType::Default,
          "Static factory method to create an OmniDSP instance.")

      .def("get_backend", &OmniDSP::get_backend)

      // Bind member functions (DSP Ops, Window Gens, Plan Factories)
      // Need to instantiate templates for float/double and complex types

      // Convolution (Real)
      .def(
          "convolve",
          [](const OmniDSP& self,
             py::array_t<float_r> input,
             py::array_t<float_r> kernel,
             ConvolutionMode mode)
          {
            auto input_vec = input.cast<std::vector<float_r>>();
            auto kernel_vec = kernel.cast<std::vector<float_r>>();
            return check_expected(
                self.convolve<float>(input_vec, kernel_vec, mode));
          },
          py::arg("input"),
          py::arg("kernel"),
          py::arg("mode"),
          "Perform 1D real convolution (float).")
      .def(
          "convolve",
          [](const OmniDSP& self,
             py::array_t<double_r> input,
             py::array_t<double_r> kernel,
             ConvolutionMode mode)
          {
            auto input_vec = input.cast<std::vector<double_r>>();
            auto kernel_vec = kernel.cast<std::vector<double_r>>();
            return check_expected(
                self.convolve<double>(input_vec, kernel_vec, mode));
          },
          py::arg("input"),
          py::arg("kernel"),
          py::arg("mode"),
          "Perform 1D real convolution (double).")

      // Convolution (Complex)
      .def(
          "convolve",
          [](const OmniDSP& self,
             py::array_t<float_c> input,
             py::array_t<float_c> kernel,
             ConvolutionMode mode)
          {
            auto input_vec = input.cast<std::vector<float_c>>();
            auto kernel_vec = kernel.cast<std::vector<float_c>>();
            return check_expected(
                self.convolve<float_c>(input_vec, kernel_vec, mode));
          },
          py::arg("input"),
          py::arg("kernel"),
          py::arg("mode"),
          "Perform 1D complex convolution (float).")
      .def(
          "convolve",
          [](const OmniDSP& self,
             py::array_t<double_c> input,
             py::array_t<double_c> kernel,
             ConvolutionMode mode)
          {
            auto input_vec = input.cast<std::vector<double_c>>();
            auto kernel_vec = kernel.cast<std::vector<double_c>>();
            return check_expected(
                self.convolve<double_c>(input_vec, kernel_vec, mode));
          },
          py::arg("input"),
          py::arg("kernel"),
          py::arg("mode"),
          "Perform 1D complex convolution (double).")

      // Correlation (Real)
      .def(
          "correlate",
          [](const OmniDSP& self,
             py::array_t<float_r> input,
             py::array_t<float_r> kernel,
             ConvolutionMode mode)
          {
            return check_expected(self.correlate<float>(
                input.cast<std::vector<float_r>>(),
                kernel.cast<std::vector<float_r>>(),
                mode));
          },
          py::arg("input"),
          py::arg("kernel"),
          py::arg("mode"),
          "Perform 1D real correlation (float).")
      .def(
          "correlate",
          [](const OmniDSP& self,
             py::array_t<double_r> input,
             py::array_t<double_r> kernel,
             ConvolutionMode mode)
          {
            return check_expected(self.correlate<double>(
                input.cast<std::vector<double_r>>(),
                kernel.cast<std::vector<double_r>>(),
                mode));
          },
          py::arg("input"),
          py::arg("kernel"),
          py::arg("mode"),
          "Perform 1D real correlation (double).")

      // Correlation (Complex)
      .def(
          "correlate",
          [](const OmniDSP& self,
             py::array_t<float_c> input,
             py::array_t<float_c> kernel,
             ConvolutionMode mode)
          {
            return check_expected(self.correlate<float_c>(
                input.cast<std::vector<float_c>>(),
                kernel.cast<std::vector<float_c>>(),
                mode));
          },
          py::arg("input"),
          py::arg("kernel"),
          py::arg("mode"),
          "Perform 1D complex correlation (float).")
      .def(
          "correlate",
          [](const OmniDSP& self,
             py::array_t<double_c> input,
             py::array_t<double_c> kernel,
             ConvolutionMode mode)
          {
            return check_expected(self.correlate<double_c>(
                input.cast<std::vector<double_c>>(),
                kernel.cast<std::vector<double_c>>(),
                mode));
          },
          py::arg("input"),
          py::arg("kernel"),
          py::arg("mode"),
          "Perform 1D complex correlation (double).")

      // One-off FFTs
      .def(
          "fft",
          [](const OmniDSP& self, py::array_t<float_c> input)
          {
            return check_expected(
                self.fft<float_c>(input.cast<std::vector<float_c>>()));
          },
          py::arg("input"),
          "Compute one-off complex FFT (float).")
      .def(
          "fft",
          [](const OmniDSP& self, py::array_t<double_c> input)
          {
            return check_expected(
                self.fft<double_c>(input.cast<std::vector<double_c>>()));
          },
          py::arg("input"),
          "Compute one-off complex FFT (double).")
      .def(
          "ifft",
          [](const OmniDSP& self, py::array_t<float_c> input)
          {
            return check_expected(
                self.ifft<float_c>(input.cast<std::vector<float_c>>()));
          },
          py::arg("input"),
          "Compute one-off complex IFFT (float, scaled by 1/N).")  // Note
                                                                   // scaling
                                                                   // difference
                                                                   // vs Plan
      .def(
          "ifft",
          [](const OmniDSP& self, py::array_t<double_c> input)
          {
            return check_expected(
                self.ifft<double_c>(input.cast<std::vector<double_c>>()));
          },
          py::arg("input"),
          "Compute one-off complex IFFT (double, scaled by 1/N).")
      .def(
          "rfft",
          [](const OmniDSP& self, py::array_t<float_r> input)
          {
            return check_expected(
                self.rfft<float>(input.cast<std::vector<float_r>>()));
          },
          py::arg("input"),
          "Compute one-off real RFFT (float).")
      .def(
          "rfft",
          [](const OmniDSP& self, py::array_t<double_r> input)
          {
            return check_expected(
                self.rfft<double>(input.cast<std::vector<double_r>>()));
          },
          py::arg("input"),
          "Compute one-off real RFFT (double).")
      .def(
          "irfft",
          [](const OmniDSP& self,
             py::array_t<float_c> input,
             size_t output_length)
          {
            return check_expected(self.irfft<float>(
                input.cast<std::vector<float_c>>(), output_length));
          },
          py::arg("input"),
          py::arg("output_length"),
          "Compute one-off real IRFFT (float, scaled by 1/N).")
      .def(
          "irfft",
          [](const OmniDSP& self,
             py::array_t<double_c> input,
             size_t output_length)
          {
            return check_expected(self.irfft<double>(
                input.cast<std::vector<double_c>>(), output_length));
          },
          py::arg("input"),
          py::arg("output_length"),
          "Compute one-off real IRFFT (double, scaled by 1/N).")

      // Window Generation
      .def(
          "bartlett_window",
          [](const OmniDSP& self, size_t length)
          { return check_expected(self.bartlett_window<float>(length)); },
          py::arg("length"),
          "Generate Bartlett window (float).")
      .def(
          "bartlett_window",
          [](const OmniDSP& self, size_t length)
          { return check_expected(self.bartlett_window<double>(length)); },
          py::arg("length"),
          "Generate Bartlett window (double).")
      // ... Add bindings for all other window types (blackman, flattop,
      // gaussian, hamming, hann, kaiser, rectangular, triangular) for float and
      // double ...
      .def(
          "kaiser_window",
          [](const OmniDSP& self, size_t length, float beta)
          { return check_expected(self.kaiser_window<float>(length, beta)); },
          py::arg("length"),
          py::arg("beta"),
          "Generate Kaiser window (float).")
      .def(
          "kaiser_window",
          [](const OmniDSP& self, size_t length, double beta)
          { return check_expected(self.kaiser_window<double>(length, beta)); },
          py::arg("length"),
          py::arg("beta"),
          "Generate Kaiser window (double).")
      // ... etc ...

      // Plan Factories (using keep_alive to tie Plan lifetime to OmniDSP
      // instance)
      .def(
          "create_fft_plan",
          [](const OmniDSP& self, size_t length)
          { return check_expected(self.create_fft_plan<float_c>(length)); },
          py::arg("length"),
          py::keep_alive<0, 1>(),
          "Create complex FFT plan (float).")
      .def(
          "create_fft_plan",
          [](const OmniDSP& self, size_t length)
          { return check_expected(self.create_fft_plan<double_c>(length)); },
          py::arg("length"),
          py::keep_alive<0, 1>(),
          "Create complex FFT plan (double).")

      .def(
          "create_rfft_plan",
          [](const OmniDSP& self, size_t length)
          { return check_expected(self.create_rfft_plan<float>(length)); },
          py::arg("length"),
          py::keep_alive<0, 1>(),
          "Create real FFT plan (float).")
      .def(
          "create_rfft_plan",
          [](const OmniDSP& self, size_t length)
          { return check_expected(self.create_rfft_plan<double>(length)); },
          py::arg("length"),
          py::keep_alive<0, 1>(),
          "Create real FFT plan (double).")

      .def(
          "create_cqt_plan",
          [](const OmniDSP& self,
             float sample_rate,
             float min_freq,
             float max_freq,
             int bins_per_octave)
          {
            // Pass 'self' as the owner pointer to the C++ factory method
            return check_expected(self.create_cqt_plan<float>(
                &self, sample_rate, min_freq, max_freq, bins_per_octave));
          },
          py::arg("sample_rate"),
          py::arg("min_freq"),
          py::arg("max_freq"),
          py::arg("bins_per_octave"),
          py::keep_alive<0, 1>(),
          "Create CQT plan (float).")
      .def(
          "create_cqt_plan",
          [](const OmniDSP& self,
             double sample_rate,
             double min_freq,
             double max_freq,
             int bins_per_octave)
          {
            return check_expected(self.create_cqt_plan<double>(
                &self, sample_rate, min_freq, max_freq, bins_per_octave));
          },
          py::arg("sample_rate"),
          py::arg("min_freq"),
          py::arg("max_freq"),
          py::arg("bins_per_octave"),
          py::keep_alive<0, 1>(),
          "Create CQT plan (double).")

      .def(
          "create_resample_plan",
          [](const OmniDSP& self,
             double input_rate,
             double output_rate,
             size_t max_input_size)
          {
            return check_expected(self.create_resample_plan<float>(
                input_rate, output_rate, max_input_size));
          },
          py::arg("input_rate"),
          py::arg("output_rate"),
          py::arg("max_input_size"),
          py::keep_alive<0, 1>(),
          "Create Resample plan (float).")
      .def(
          "create_resample_plan",
          [](const OmniDSP& self,
             double input_rate,
             double output_rate,
             size_t max_input_size)
          {
            return check_expected(self.create_resample_plan<double>(
                input_rate, output_rate, max_input_size));
          },
          py::arg("input_rate"),
          py::arg("output_rate"),
          py::arg("max_input_size"),
          py::keep_alive<0, 1>(),
          "Create Resample plan (double).")

      // Add create_convolution_plan / create_correlation_plan bindings...
      // Example:
      .def(
          "create_convolution_plan",
          [](const OmniDSP& self,
             py::array_t<float> kernel,
             ConvolutionMode mode)
          {
            return check_expected(self.create_convolution_plan<float>(
                kernel.cast<std::vector<float>>(), mode));
          },
          py::arg("kernel"),
          py::arg("mode"),
          py::keep_alive<0, 1>(),
          "Create Convolution plan (float).")
      // ... add other types ...

      ;  // End OmniDSP class binding
}

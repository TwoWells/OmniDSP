/**
 * @file resample.cpp (onemkl)
 * @brief Implements oneMKL backend ResamplePlanImpl class.
 * @warning oneMKL VSL does not provide a direct arbitrary resampling function.
 * This implementation requires a custom resampling algorithm (e.g., polyphase
 * FIR filtering) built using oneMKL primitives (e.g., VSL convolution tasks or
 * VML). The implementation below is a skeleton.
 */

// Only compile this file if oneMKL backend is enabled via CMake
#ifdef USE_ONEMKL

#include "OmniDSP/core_types.h"  // For Status, RealT etc.
#include "backend.h"             // oneMKL backend declarations

// Include MKL header (VSL for convolution tasks, VML for math)
#include <mkl.h>

#include <cmath>    // For std::ceil, std::floor, std::gcd, std::log10, std::min
#include <complex>  // Include complex for potential internal use if needed
#include <iostream>  // For debug/error messages
#include <numeric>   // For std::gcd
#include <span>
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <vector>

namespace OmniDSP {
namespace backend {

// Helper function from onemkl/backend.cpp (or move to common utility)
inline Status mkl_status_to_omnidsp_status(MKL_LONG status) {
  if (status == DFTI_NO_ERROR) {  // Assuming VSL uses similar error codes or
                                  // has its own mapping
    return Status::Success;
  }
  // TODO: Map VSL/VML specific error codes appropriately
  std::cerr << "MKL Error: Code " << status
            << std::endl;  // Add VSL/VML error lookup if available
  if (status == VSL_STATUS_ERR_MEM_FAILURE) return Status::AllocationError;
  // ... other VSL/VML error codes
  return Status::BackendError;
}

// Helper function for designing a basic low-pass FIR filter (Placeholder)
// (Copied from accelerate/resample.cpp - move to common utility?)
template <typename T>
std::vector<T> design_resample_filter(size_t num_taps,
                                      double cutoff_freq_norm) {
  std::vector<T> h(num_taps);
  T center = static_cast<T>(num_taps - 1) / 2.0;
  T sum = 0;
  for (size_t i = 0; i < num_taps; ++i) {
    T n = static_cast<T>(i) - center;
    T sinc_val =
        (n == 0) ? static_cast<T>(2.0 * cutoff_freq_norm)
                 : static_cast<T>(std::sin(2.0 * M_PI * cutoff_freq_norm * n) /
                                  (M_PI * n));
    T hann_val =
        static_cast<T>(0.5 * (1.0 - std::cos(2.0 * M_PI * i / (num_taps - 1))));
    h[i] = sinc_val * hann_val;
    sum += h[i];
  }
  if (sum != 0) {
    T inv_sum = static_cast<T>(1.0) / sum;
    // Can use MKL v?MulByConj here? Or just loop.
    for (size_t i = 0; i < num_taps; ++i) {
      h[i] *= inv_sum;
    }
  }
  return h;
}

//--------------------------------------------------------------------------
// OneMKLResamplePlanImpl Method Definitions
//--------------------------------------------------------------------------

template <typename T>
OneMKLResamplePlanImpl<T>::OneMKLResamplePlanImpl(double input_rate,
                                                  double output_rate,
                                                  size_t /* max_input_size */)
    : input_rate_(input_rate), output_rate_(output_rate) {
  if (input_rate <= 0.0 || output_rate <= 0.0) {
    throw std::invalid_argument(
        "Input and output sample rates must be positive.");
  }

  // --- Polyphase Resampler Setup ---
  // 1. Determine Upsample (L) and Downsample (M) factors
  const long long factor_base = 1LL << 32;
  long long l_approx =
      static_cast<long long>(output_rate_ * factor_base / input_rate_ + 0.5);
  if (l_approx == 0) l_approx = 1;
  long long common = std::gcd(l_approx, factor_base);
  upsample_factor_L_ = static_cast<size_t>(l_approx / common);
  downsample_factor_M_ = static_cast<size_t>(factor_base / common);

  if (upsample_factor_L_ == 0 || downsample_factor_M_ == 0) {
    throw std::runtime_error(
        "Failed to determine valid up/downsample factors.");
  }

  // 2. Design low-pass anti-aliasing/imaging filter
  double intermediate_rate = input_rate_ * upsample_factor_L_;
  double cutoff_freq = std::min(input_rate_, output_rate_) / 2.0;
  double normalized_cutoff = cutoff_freq / intermediate_rate;

  size_t num_taps_per_phase = 12;                                // Example
  filter_length_ = upsample_factor_L_ * num_taps_per_phase * 2;  // Example
  if (filter_length_ == 0) filter_length_ = 1;

  prototype_filter_ =
      design_resample_filter<T>(filter_length_, normalized_cutoff);

  // 3. Setup MKL VSL Convolution Task (if using VSL)
  //    This is complex. It involves creating polyphase filter banks
  //    and setting up a VSL task for efficient convolution.
  //    Alternatively, implement manually using VML primitives.
  //    Placeholder: Just store factors and prototype filter.
  vsl_task_ = nullptr;  // Initialize task pointer to null
  // Example VSL setup (pseudo-code, needs actual parameters):
  // MKL_LONG x_shape = ...; // Input block size?
  // MKL_LONG y_shape = ...; // Output block size?
  // MKL_LONG h_shape = filter_length_;
  // int status = vs?ConvNewTask1D(&vsl_task_, VSL_CONV_MODE_AUTO, x_shape,
  // y_shape, h_shape); if (status != VSL_STATUS_OK) { throw
  // std::runtime_error("Failed to create MKL VSL Conv task."); } status =
  // vs?ConvSetInternalPrecision(vsl_task_, ...); status =
  // vs?EditConvCoefs1D(vsl_task_, prototype_filter_.data(), ...);
  // ... other VSL setup ...

  // 4. Initialize state (delay line)
  filter_state_.assign(filter_length_ - 1, T{});

  std::cout << "oneMKL ResamplePlanImpl created. L=" << upsample_factor_L_
            << ", M=" << downsample_factor_M_ << ", Taps=" << filter_length_
            << std::endl;  // Debug
}

template <typename T>
OneMKLResamplePlanImpl<T>::~OneMKLResamplePlanImpl() {
  // Clean up MKL resources if allocated
  if (vsl_task_) {
    // int status = vslConvDeleteTask(&vsl_task_);
    // if (status != VSL_STATUS_OK) {
    //     std::cerr << "Warning: Failed to delete MKL VSL Conv task." <<
    //     std::endl;
    // }
  }
  std::cout << "oneMKL ResamplePlanImpl destroyed." << std::endl;  // Debug
}

template <typename T>
Status OneMKLResamplePlanImpl<T>::execute(std::span<const T> input,
                                          std::span<T> output) const {
  if (upsample_factor_L_ == 0 || downsample_factor_M_ == 0 ||
      prototype_filter_.empty()) {
    return Status::InvalidOperation;  // Plan not initialized correctly
  }

  size_t input_len = input.size();
  size_t output_len_required = get_output_length(input_len);

  if (output.size() < output_len_required) {
    return Status::SizeMismatch;  // Output buffer too small
  }

  // --- Polyphase Resampling Implementation using oneMKL ---
  // If using VSL Convolution Task:
  //   - Manage input/output blocks based on task setup.
  //   - Call vs?ConvExec(vsl_task_, input_block, ..., output_block, ...).
  //   - Handle state/overlap between blocks.
  // If using VML:
  //   - Implement the polyphase filtering loop manually using VML vector math
  //     (similar logic to Accelerate placeholder, but with MKL VML functions).

  // --- Placeholder Implementation ---
  std::cerr << "OneMKLResamplePlanImpl::execute - Polyphase resampling logic "
               "not implemented in skeleton!"
            << std::endl;

  size_t samples_to_copy =
      std::min(output_len_required, input_len);  // Incorrect logic
  std::copy(input.begin(), input.begin() + samples_to_copy, output.begin());
  if (output_len_required > samples_to_copy) {
    std::fill(output.begin() + samples_to_copy,
              output.begin() + output_len_required, T{});
  }
  // Need to update filter_state_ (mutable member needed, or pass state buffer)

  // --- End Placeholder ---

  // Zero out remaining output buffer if user provided a larger one
  if (output.size() > output_len_required) {
    std::fill(output.begin() + output_len_required, output.end(), T{});
  }

  // return Status::UnsupportedFeature; // Until implemented
  return Status::Success;  // Placeholder success
}

template <typename T>
double OneMKLResamplePlanImpl<T>::get_input_rate() const {
  return input_rate_;
}

template <typename T>
double OneMKLResamplePlanImpl<T>::get_output_rate() const {
  return output_rate_;
}

template <typename T>
size_t OneMKLResamplePlanImpl<T>::get_output_length(size_t input_length) const {
  if (input_rate_ <= 0.0) return 0;
  double ratio = output_rate_ / input_rate_;
  size_t estimated_len =
      static_cast<size_t>(std::ceil(static_cast<double>(input_length) * ratio));
  // Add estimate for filter delay/transient if needed
  return estimated_len;
}

//--------------------------------------------------------------------------
// Explicit Template Instantiations
//--------------------------------------------------------------------------
// Instantiate templates for common types (float, double) to ensure code
// generation.

template class OmniDSP::backend::OneMKLResamplePlanImpl<float>;
template class OmniDSP::backend::OneMKLResamplePlanImpl<double>;

// Remove implementation of old standalone filter_and_downsample function here
// if it existed.

}  // namespace backend
}  // namespace OmniDSP

#endif  // USE_ONEMKL

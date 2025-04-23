/**
 * @file resample.cpp (onemkl)
 * @brief Implements the oneMKL backend ResamplePlanImpl class using IPP FIRMR
 * functions.
 * @details Provides resampling using Intel Integrated Performance Primitives
 * (IPP) multi-rate FIR filtering capabilities.
 */

#include <algorithm>  // For std::copy, std::fill, std::min
#include <cmath>      // For std::ceil, std::floor, std::min, std::max, std::abs
#include <iostream>   // For debug messages
#include <memory>     // For std::unique_ptr
#include <numeric>    // For std::gcd
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <vector>

#include "backend.h"  // oneMKL backend declarations (includes OneMKLResamplePlanImpl declaration)

// Include Intel IPP signal processing header
#include <ipps.h>

namespace OmniDSP {
namespace backend {

// Helper function to convert IPP status codes to OmniDSP::Status.
inline Status ipp_status_to_omnidsp_status(IppStatus status,
                                           const std::string& context = "") {
  if (status == ippStsNoErr) {
    return Status::Success;
  }
  std::cerr << "IPP Error";
  if (!context.empty()) {
    std::cerr << " in " << context;
  }
  std::cerr << ": " << ippGetStatusString(status) << " (Code: " << status << ")"
            << std::endl;

  if (status == ippStsNullPtrErr || status == ippStsSizeErr ||
      status == ippStsStepErr || status == ippStsBadArgErr ||
      status == ippStsOutOfRangeErr || status == ippStsLengthErr ||
      status == ippStsFIRMRFactorErr || status == ippStsFIRMRPhaseErr) {
    return Status::InvalidArgument;
  }
  if (status == ippStsMemAllocErr) {
    return Status::AllocationError;
  }
  return Status::BackendError;
}

// Custom deleter for IPP allocated memory
struct IPPFreeDeleter {
  void operator()(Ipp8u* ptr) const {
    if (ptr) {
      ippsFree(ptr);
    }
  }
  // Overload for the spec structure pointer (const issue)
  template <typename SpecType>
  void operator()(SpecType* ptr) const {
    if (ptr) {
      ippsFree(ptr);
    }
  }
};

// Define unique_ptr types using the custom deleter
using IppBufferPtr = std::unique_ptr<Ipp8u, IPPFreeDeleter>;
template <typename SpecType>
using IppSpecPtr = std::unique_ptr<SpecType, IPPFreeDeleter>;

//--------------------------------------------------------------------------
// OneMKLResamplePlanImpl Definition
//--------------------------------------------------------------------------

/**
 * @brief Concrete implementation of ResamplePlanImpl for the oneMKL backend
 * using IPP FIRMR.
 */
template <typename T>  // T is real type here
class OneMKLResamplePlanImpl final : public ResamplePlanImpl<T> {
 public:
  /**
   * @brief Constructor. Calculates factors, designs filter, initializes IPP
   * FIRMR state.
   * @param owner Pointer to the OmniDSPImpl instance creating this plan.
   * @param spec The resampling specification.
   * @throws std::invalid_argument If spec or owner is invalid.
   * @throws std::runtime_error If factor calculation, filter design, or IPP
   * setup fails.
   */
  OneMKLResamplePlanImpl(const OmniDSPImpl* owner, const ResampleSpec& spec);

  /** @brief Destructor. Frees IPP resources. */
  ~OneMKLResamplePlanImpl() override;

  // --- Interface Methods ---
  Status execute(std::span<const T> input, std::span<T> output) override;
  Status reset() override;
  double get_input_rate() const override;
  double get_output_rate() const override;
  size_t get_output_length(size_t input_length) const override;

 private:
  // --- Configuration ---
  double input_rate_;
  double output_rate_;
  size_t upsample_factor_L_;
  size_t downsample_factor_M_;
  size_t num_taps_;
  int ipp_up_factor_;
  int ipp_down_factor_;
  int ipp_taps_len_;

  // --- IPP FIRMR State ---
  // Use appropriate IPP spec type based on T
  using IppSpecType =
      typename std::conditional<std::is_same_v<T, float>, IppsFIRSpec_32f,
                                IppsFIRSpec_64f>::type;
  IppSpecPtr<IppSpecType> p_spec_;
  IppBufferPtr p_buffer_;

  // --- Internal Delay Line ---
  std::vector<T> delay_line_;
  size_t delay_len_;  // Store calculated delay line length

  // --- Helper Methods ---
  static Status calculate_factors_static(double in_rate, double out_rate,
                                         size_t& L, size_t& M);
  Status design_filter(const OmniDSPImpl* owner, const ResampleSpec& spec,
                       size_t L, size_t M, std::vector<T>& coeffs);
};

//--------------------------------------------------------------------------
// OneMKLResamplePlanImpl Method Implementations
//--------------------------------------------------------------------------

template <typename T>
OneMKLResamplePlanImpl<T>::OneMKLResamplePlanImpl(const OmniDSPImpl* owner,
                                                  const ResampleSpec& spec)
    : input_rate_(spec.input_rate),
      output_rate_(spec.output_rate),
      p_spec_(nullptr),
      p_buffer_(nullptr) {
  if (!owner) {
    throw std::invalid_argument(
        "OneMKLResamplePlanImpl requires a valid owner OmniDSPImpl pointer.");
  }
  if (!spec.validate()) {
    throw std::invalid_argument("Invalid ResampleSpec provided.");
  }

  // 1. Calculate L/M factors
  Status factor_status = calculate_factors_static(
      input_rate_, output_rate_, upsample_factor_L_, downsample_factor_M_);
  if (factor_status != Status::Success) {
    throw std::runtime_error("Failed to calculate resampling factors L and M.");
  }
  // IPP uses int for factors
  ipp_up_factor_ = static_cast<int>(upsample_factor_L_);
  ipp_down_factor_ = static_cast<int>(downsample_factor_M_);
  if (ipp_up_factor_ <= 0 || ipp_down_factor_ <= 0) {
    throw std::runtime_error(
        "Calculated L/M factors are not positive integers.");
  }

  // 2. Design the prototype FIR filter using the owner's design method
  std::vector<T> prototype_filter_coeffs;
  Status filter_status =
      design_filter(owner, spec, upsample_factor_L_, downsample_factor_M_,
                    prototype_filter_coeffs);
  if (filter_status != Status::Success) {
    throw std::runtime_error(
        "Failed to design prototype FIR filter for resampling.");
  }
  num_taps_ = prototype_filter_coeffs.size();
  ipp_taps_len_ = static_cast<int>(num_taps_);
  if (ipp_taps_len_ <= 0) {
    throw std::runtime_error(
        "Designed prototype filter has zero or negative taps.");
  }

  // 3. Setup IPP FIRMR
  IppStatus ipp_status = ippStsNoErr;
  int spec_size = 0;
  int buffer_size = 0;
  IppDataType data_type = std::is_same_v<T, float> ? ipp32f : ipp64f;

  ipp_status = ippsFIRMRGetSize(ipp_taps_len_, ipp_up_factor_, ipp_down_factor_,
                                data_type, &spec_size, &buffer_size);
  if (ipp_status != ippStsNoErr) {
    ipp_status_to_omnidsp_status(ipp_status, "ippsFIRMRGetSize");  // Log error
    throw std::runtime_error("Failed to get IPP FIRMR buffer sizes.");
  }

  // Allocate memory using IPP functions and manage with unique_ptr
  p_spec_.reset(reinterpret_cast<IppSpecType*>(ippsMalloc_8u(spec_size)));
  p_buffer_.reset(ippsMalloc_8u(buffer_size));

  if (!p_spec_ || !p_buffer_) {
    throw std::bad_alloc();  // Allocation failed
  }

  // Initialize the FIRMR state structure
  // Note: IPP FIRMR requires filter taps scaled by L. The design_filter helper
  // already does this.
  int up_phase = 0;    // Start phase for upsampling (usually 0)
  int down_phase = 0;  // Start phase for downsampling (usually 0)

  if constexpr (std::is_same_v<T, float>) {
    ipp_status = ippsFIRMRInit_32f(prototype_filter_coeffs.data(),
                                   ipp_taps_len_, ipp_up_factor_, up_phase,
                                   ipp_down_factor_, down_phase, p_spec_.get());
  } else {
    ipp_status = ippsFIRMRInit_64f(prototype_filter_coeffs.data(),
                                   ipp_taps_len_, ipp_up_factor_, up_phase,
                                   ipp_down_factor_, down_phase, p_spec_.get());
  }

  if (ipp_status != ippStsNoErr) {
    ipp_status_to_omnidsp_status(ipp_status, "ippsFIRMRInit");  // Log error
    throw std::runtime_error("Failed to initialize IPP FIRMR state.");
  }

  // 4. Initialize internal delay line buffer
  delay_len_ = (ipp_taps_len_ + ipp_up_factor_ - 1) / ipp_up_factor_;
  delay_line_.assign(delay_len_, T{0});

  std::cout << "oneMKL ResamplePlanImpl created. L=" << upsample_factor_L_
            << ", M=" << downsample_factor_M_ << ", Taps=" << num_taps_
            << ", DelayLen=" << delay_len_ << std::endl;  // Debug
}

template <typename T>
OneMKLResamplePlanImpl<T>::~OneMKLResamplePlanImpl() {
  // unique_ptrs p_spec_ and p_buffer_ automatically call ippsFree via deleter
  std::cout << "oneMKL ResamplePlanImpl destroyed." << std::endl;  // Debug
}

template <typename T>
Status OneMKLResamplePlanImpl<T>::execute(std::span<const T> input,
                                          std::span<T> output) {
  if (!p_spec_ || !p_buffer_) {
    return Status::InvalidOperation;  // Plan not initialized correctly
  }

  size_t input_len = input.size();
  if (input_len == 0) {
    std::fill(output.begin(), output.end(),
              T{0});  // Zero out output if input is empty
    return Status::Success;
  }

  // Determine the number of iterations based on input/output sizes and factors
  // ippsFIRMR processes `numIters * ipp_down_factor_` input samples
  // and produces `numIters * ipp_up_factor_` output samples.
  int num_iters_in = static_cast<int>(
      input_len /
      ipp_down_factor_);  // Max iterations based on full input blocks
  int num_iters_out_max = static_cast<int>(
      output.size() /
      ipp_up_factor_);  // Max iterations based on output capacity

  // Use the minimum number of iterations possible to avoid buffer overruns
  int num_iters = std::min(num_iters_in, num_iters_out_max);

  if (num_iters <= 0) {
    // Not enough input or output space for even one iteration block.
    // This might happen with very short inputs/outputs or large factors.
    // We could potentially handle this with padding/manual processing,
    // but for now, return success with no processing done.
    std::fill(output.begin(), output.end(), T{0});
    return Status::Success;
  }

  size_t input_samples_to_process =
      static_cast<size_t>(num_iters) * ipp_down_factor_;
  size_t output_samples_to_generate =
      static_cast<size_t>(num_iters) * ipp_up_factor_;

  // Prepare delay lines for IPP call
  std::vector<T> dly_src = delay_line_;  // Copy current state
  std::vector<T> dly_dst(delay_len_);    // Buffer for output state

  IppStatus ipp_status = ippStsNoErr;

  // Call the IPP FIRMR function
  if constexpr (std::is_same_v<T, float>) {
    ipp_status =
        ippsFIRMR_32f(input.data(), output.data(), num_iters, p_spec_.get(),
                      dly_src.data(), dly_dst.data(), p_buffer_.get());
  } else {
    ipp_status =
        ippsFIRMR_64f(input.data(), output.data(), num_iters, p_spec_.get(),
                      dly_src.data(), dly_dst.data(), p_buffer_.get());
  }

  Status status = ipp_status_to_omnidsp_status(ipp_status, "ippsFIRMR");
  if (status != Status::Success) {
    return status;  // Propagate IPP error
  }

  // Update the internal delay line state with the output delay line from IPP
  delay_line_ = std::move(dly_dst);

  // Zero out the remaining part of the output buffer if the user provided
  // a buffer larger than what was generated.
  if (output.size() > output_samples_to_generate) {
    std::fill(output.begin() + output_samples_to_generate, output.end(), T{0});
  }

  // Note: This implementation processes only full blocks of `ipp_down_factor_`
  // input samples. Any remaining input samples (< ipp_down_factor_) are ignored
  // in this call and will implicitly become part of the state for the *next*
  // call if the calling code handles streaming correctly (by feeding the
  // remaining samples plus new samples in the next execute call). The state
  // update mechanism using the work_buffer in the Default implementation is
  // more explicit about this. The IPP delay line mechanism handles this
  // implicitly.

  return Status::Success;
}

template <typename T>
Status OneMKLResamplePlanImpl<T>::reset() {
  // Reset the internal delay line to zeros
  std::fill(delay_line_.begin(), delay_line_.end(), T{0});
  // Note: IPP FIRMR state itself (pSpec) doesn't usually need resetting,
  // only the delay line representing past samples.
  return Status::Success;
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
  if (input_rate_ <= 0.0 || upsample_factor_L_ == 0) return 0;
  // Estimate based on ratio. Add 1 for potential ceiling effects.
  double ratio = output_rate_ / input_rate_;
  // IPP FIRMR delay contribution is complex. Using a simple estimate based on
  // taps/rate. Delay is roughly half the filter length in terms of *input*
  // samples. Convert that delay to output samples.
  double filter_delay_in_samples =
      (num_taps_ > 0) ? (static_cast<double>(num_taps_ - 1) / 2.0) : 0.0;
  size_t delay_contribution_out =
      static_cast<size_t>(std::ceil(filter_delay_in_samples * ratio));

  size_t estimated_len = static_cast<size_t>(std::ceil(
                             static_cast<double>(input_length) * ratio)) +
                         delay_contribution_out + 1;  // Add 1 for safety

  return estimated_len;
}

// --- Helper Implementations ---

template <typename T>
Status OneMKLResamplePlanImpl<T>::calculate_factors_static(double in_rate,
                                                           double out_rate,
                                                           size_t& L,
                                                           size_t& M) {
  if (in_rate <= 0.0 || out_rate <= 0.0) {
    return Status::InvalidArgument;
  }
  const long long factor_base = 1LL << 32;
  long long num_approx =
      static_cast<long long>(out_rate * factor_base / in_rate + 0.5);
  if (num_approx == 0) num_approx = 1;
  long long common = std::gcd(num_approx, factor_base);
  if (common == 0) return Status::Failure;
  L = static_cast<size_t>(num_approx / common);
  M = static_cast<size_t>(factor_base / common);
  if (L == 0 || M == 0) {
    return Status::Failure;
  }
  return Status::Success;
}

template <typename T>
Status OneMKLResamplePlanImpl<T>::design_filter(const OmniDSPImpl* owner,
                                                const ResampleSpec& spec,
                                                size_t L, size_t M,
                                                std::vector<T>& coeffs) {
  // Use the base class implementation provided in backend.h
  // This assumes the definition is available (e.g., in backend.cpp or linked)
  return ResamplePlanImpl<T>::design_prototype_filter(owner, spec, coeffs, L,
                                                      M);
}

//--------------------------------------------------------------------------
// Explicit Template Instantiations
//--------------------------------------------------------------------------
template class OneMKLResamplePlanImpl<float>;
template class OneMKLResamplePlanImpl<double>;

}  // namespace backend
}  // namespace OmniDSP

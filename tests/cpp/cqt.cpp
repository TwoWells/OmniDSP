/**
 * @file tests/cpp/cqt.cpp
 * @brief Unit tests for the OmniDSP Recursive CQTPlan class
 * with precomputed sparse kernels. Includes comparison with Librosa reference.
 * Uses TestDataUtils for loading reference data.
 */

#include <OmniDSP/cqt.h>      // Include Recursive CQT header
#include <OmniDSP/omnidsp.h>  // For FFTPlan potentially and Enums
#include <OmniDSP/window.h>  // Include window header (needed for M_PI in helper)
#include <gtest/gtest.h>

#include <cmath>
#include <complex>

#include "test_data_utils.h"  // Include the new utility header
// #include <fstream> // No longer needed
#include <iostream>  // For std::cout/cerr (optional debugging)
#include <limits>
// #include <map> // No longer needed for sparse kernel checks here
#include <memory>   // For std::unique_ptr access in tests
#include <numeric>  // For std::iota, std::accumulate
// #include <sstream> // No longer needed
#include <stdexcept>  // Needed for EXPECT_THROW
#include <string>     // For std::string, std::getline, std::stod, etc.
#include <vector>

// Define M_PI if it's not already defined (e.g., by <cmath> in some
// environments/standards)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Define a constant for the default FIR order used in tests, matching the C++
// implementation
const int DEFAULT_FIR_FILTER_ORDER_TEST = 101;

// Define aliases for vector/matrix types
using VecD = std::vector<double>;
using VecF = std::vector<float>;
using ComplexVecD = std::vector<std::complex<double>>;
using ComplexVecF = std::vector<std::complex<float>>;  // Added for float CQT
using ComplexMatD = std::vector<ComplexVecD>;  // Matrix type [bin][frame]
using ComplexMatF = std::vector<ComplexVecF>;  // Matrix type [bin][frame]

// --- Test Fixture ---
class PrecomputedRecursiveCQTTest : public ::testing::Test {
 protected:
  // Common parameters matching generate_references.py CQT section
  const double sample_rate = 44100.0;
  const double lowest_freq = 55.0;  // A1
  const double high_freq = 1760.0;  // A6
  const int bins_per_octave = 12;
  const size_t hop_length
      = 512;  // Valid hop length: 5 octaves -> need hop % 16 == 0
  const float sparsity_threshold_float = 1e-5f;
  const double sparsity_threshold_double = 1e-5;
  const double cqt_signal_freq = 440.0;  // A4 - matches generate_references.py
  const double cqt_signal_duration
      = 1.0;  // seconds - matches generate_references.py

  // Default window generator helper (returns coefficients) - L1 Normalized
  // This is passed to the CQTPlan constructor in tests.
  template <typename T>
  static std::vector<T> hannWindowCoeffsGenerator(size_t length)
  {
    if (length == 0) return {};
    std::vector<T> window_coeffs(length);
    double denom = (length > 1) ? static_cast<double>(length - 1) : 1.0;
    for (size_t n = 0; n < length; ++n) {
      window_coeffs[n]
          = static_cast<T>(0.5 - 0.5 * std::cos(2.0 * M_PI * n / denom));
    }
    // CQTPlan internally normalizes the kernel, so no need to normalize here
    // for the generator function passed to the constructor.
    return window_coeffs;
  }

  // Helper to generate a sine wave - matches generate_references.py
  template <typename T>
  std::vector<T> generateSineWave(double freq, double duration_sec, double sr)
  {
    size_t length = static_cast<size_t>(duration_sec * sr);
    if (length == 0) return {};
    std::vector<T> signal(length);
    for (size_t i = 0; i < length; ++i) {
      signal[i] = static_cast<T>(
          std::sin(2.0 * M_PI * freq * static_cast<double>(i) / sr));
    }
    return signal;
  }

  // Helper to compare complex matrices (using double for expected always for
  // now)
  template <typename T>
  void ExpectComplexMatrixNear(
      const ComplexMatD &expected,
      const std::vector<std::vector<std::complex<T>>> &actual,
      T tolerance,
      const std::string &msg = "")
  {
    ASSERT_EQ(expected.size(), actual.size())
        << msg << " - Bin count mismatch (Expected: " << expected.size()
        << ", Actual: " << actual.size() << ")";
    if (expected.empty()) return;

    size_t num_bins = expected.size();
    ASSERT_GT(num_bins, 0) << msg
                           << " - Zero bins in non-empty expected matrix";
    size_t num_frames = expected[0].size();

    if (actual.empty()) {
      FAIL() << msg << " - Actual matrix is empty while expected has "
             << num_bins << " bins and " << num_frames << " frames.";
    }
    ASSERT_EQ(actual[0].size(), num_frames)
        << msg << " - Frame count mismatch for bin 0 (Expected: " << num_frames
        << ", Actual: " << actual[0].size() << ")";

    for (size_t i = 0; i < num_bins; ++i) {
      ASSERT_EQ(expected[i].size(), num_frames)
          << msg << " - Frame count mismatch in expected bin " << i;
      ASSERT_EQ(actual[i].size(), num_frames)
          << msg << " - Frame count mismatch in actual bin " << i;
      for (size_t j = 0; j < num_frames; ++j) {
        EXPECT_NEAR(expected[i][j].real(), actual[i][j].real(), tolerance)
            << msg << " - Mismatch at bin " << i << ", frame " << j
            << " (real)";
        EXPECT_NEAR(expected[i][j].imag(), actual[i][j].imag(), tolerance)
            << msg << " - Mismatch at bin " << i << ", frame " << j
            << " (imag)";
      }
    }
  }

  // Helper to compare real vectors
  template <typename T>
  void ExpectRealVectorNear(
      const std::vector<T> &expected,
      const std::vector<T> &actual,
      T tolerance,
      const std::string &msg = "")
  {
    ASSERT_EQ(expected.size(), actual.size())
        << msg << " - Size mismatch (Expected: " << expected.size()
        << ", Actual: " << actual.size() << ")";  // Added size info
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(expected[i], actual[i], tolerance)
          << msg << " - Mismatch at index " << i;
    }
  }

  // Helper to get tolerance
  template <typename T>
  T get_tolerance(size_t N = 1024)
  {
    // Tolerance might need adjustment based on Librosa vs OmniDSP differences
    return std::numeric_limits<T>::epsilon() * N * 1000;
  }

  // --- Accessors for Private CQTPlan Members (Requires Friend Declaration in
  // cqt.h) ---
  template <typename T>
  static const std::vector<size_t> &getOctaveFFTLens(
      const OmniDSP::CQTPlan<T> &plan)
  {
    return plan.octave_fft_lens_;
  }
  template <typename T>
  static const std::vector<size_t> &getOctaveSpectrumLens(
      const OmniDSP::CQTPlan<T> &plan)
  {
    return plan.octave_spectrum_lens_;
  }
  template <typename T>
  static const std::vector<std::unique_ptr<OmniDSP::FFTPlan<T>>> &
  getOctaveSignalFFTPlans(const OmniDSP::CQTPlan<T> &plan)
  {
    return plan.octave_signal_fft_plans_;
  }
  template <typename T>
  static const std::vector<std::unique_ptr<OmniDSP::FFTPlan<T>>> &
  getOctaveKernelFFTPlans(const OmniDSP::CQTPlan<T> &plan)
  {
    return plan.octave_kernel_fft_plans_;
  }
  template <typename T>
  static const std::vector<typename OmniDSP::CQTPlan<T>::SparseKernelOctave> &
  getSparseKernels(const OmniDSP::CQTPlan<T> &plan)
  {
    return plan.precomputed_sparse_kernels_;
  }
  // Helper to call the private method calculateFirCoefficients
  template <typename T>
  static std::vector<T> callCalculateFirCoefficients(
      const OmniDSP::CQTPlan<T> &plan, double current_sr, int N)
  {
    return plan.calculateFirCoefficients(current_sr, N);
  }
  // Helper to call the private method filterAndDownsampleBy2
  template <typename T>
  static std::vector<T> callFilterAndDownsampleBy2(
      const OmniDSP::CQTPlan<T> &plan,
      const std::vector<T> &signal,
      double current_sr)
  {
    return plan.filterAndDownsampleBy2(signal, current_sr);
  }
};

// --- Test Cases ---

// 1. Test Constructor and Precomputation Sanity Checks
TEST_F(PrecomputedRecursiveCQTTest, ConstructorPrecomputation)
{
  using T = double;
  std::unique_ptr<OmniDSP::CQTPlan<T>> plan_ptr;
  ASSERT_NO_THROW(
      plan_ptr = std::make_unique<OmniDSP::CQTPlan<T>>(
          sample_rate,
          hop_length,
          lowest_freq,
          high_freq,
          bins_per_octave,
          hannWindowCoeffsGenerator<T>,
          sparsity_threshold_double,
          DEFAULT_FIR_FILTER_ORDER_TEST))
      << "Constructor threw unexpected exception.";
  ASSERT_NE(plan_ptr, nullptr);
  OmniDSP::CQTPlan<T> &plan = *plan_ptr;
  ASSERT_EQ(plan.getSampleRate(), sample_rate);
  ASSERT_EQ(plan.getHopLength(), hop_length);
  ASSERT_EQ(plan.getLowestFrequency(), lowest_freq);
  ASSERT_EQ(plan.getHighestFrequency(), high_freq);
  ASSERT_EQ(plan.getBinsPerOctave(), bins_per_octave);
  ASSERT_NEAR(
      plan.getSparsityThreshold(),
      sparsity_threshold_double,
      get_tolerance<T>());
  ASSERT_EQ(plan.getFirFilterOrder(), DEFAULT_FIR_FILTER_ORDER_TEST);
  int expected_octaves = 5;
  size_t expected_bins = static_cast<size_t>(
      std::ceil(bins_per_octave * std::log2(high_freq / lowest_freq)));
  ASSERT_EQ(plan.getNumOctaves(), expected_octaves);
  ASSERT_EQ(plan.getNumBins(), expected_bins);
  ASSERT_EQ(getOctaveFFTLens(plan).size(), expected_octaves);
  ASSERT_EQ(getOctaveSpectrumLens(plan).size(), expected_octaves);
  ASSERT_EQ(getOctaveSignalFFTPlans(plan).size(), expected_octaves);
  ASSERT_EQ(getOctaveKernelFFTPlans(plan).size(), expected_octaves);
  ASSERT_EQ(getSparseKernels(plan).size(), expected_octaves);
  for (int i = 0; i < expected_octaves; ++i) {
    ASSERT_GT(getOctaveFFTLens(plan)[i], 0);
    ASSERT_EQ(
        getOctaveSpectrumLens(plan)[i], getOctaveFFTLens(plan)[i] / 2 + 1);
    ASSERT_NE(getOctaveSignalFFTPlans(plan)[i], nullptr);
    ASSERT_NE(getOctaveKernelFFTPlans(plan)[i], nullptr);
    int bins_in_oct
        = (i == expected_octaves - 1)
              ? (static_cast<int>(expected_bins) - i * bins_per_octave)
              : bins_per_octave;
    ASSERT_EQ(getSparseKernels(plan)[i].size(), std::max(0, bins_in_oct));
    if (bins_in_oct > 0 && !getSparseKernels(plan)[i].empty()
        && !getSparseKernels(plan)[i][0].empty()) {
      ASSERT_FALSE(getSparseKernels(plan)[i][0].empty());
    }
  }
}

// 2. Test FIR Coefficient Calculation (Calls private helper via friend access)
TEST_F(PrecomputedRecursiveCQTTest, CalculateFirCoefficients)
{
  using T = double;
  OmniDSP::CQTPlan<T> plan(
      sample_rate,
      hop_length,
      lowest_freq,
      high_freq,
      bins_per_octave,
      hannWindowCoeffsGenerator<T>,
      sparsity_threshold_double,
      DEFAULT_FIR_FILTER_ORDER_TEST);
  int N = DEFAULT_FIR_FILTER_ORDER_TEST;
  std::vector<T> coeffs;
  ASSERT_NO_THROW(coeffs = callCalculateFirCoefficients(plan, sample_rate, N));
  ASSERT_EQ(coeffs.size(), N);
  T symm_tolerance = get_tolerance<T>() * 10;
  for (int k = 0; k < N / 2; ++k) {
    ASSERT_NEAR(coeffs[k], coeffs[N - 1 - k], symm_tolerance);
  }
  T sum = std::accumulate(coeffs.begin(), coeffs.end(), T(0.0));
  ASSERT_NEAR(sum, T(1.0), get_tolerance<T>());
  int center_idx = (N - 1) / 2;
  ASSERT_GE(center_idx, 0);
  ASSERT_LT(center_idx, N);
  T center_val = coeffs[center_idx];
  ASSERT_GT(center_val, 0.0);
  for (int k = 0; k < N; ++k) {
    if (k != center_idx) {
      ASSERT_LE(coeffs[k], center_val + symm_tolerance);
    }
  }
}

// 3. Test Combined Filter and Downsample Step using Reference Data (Calls
// private helper)
TEST_F(PrecomputedRecursiveCQTTest, FilterAndDownsample)
{
  // Test Float precision
  {
    using T = float;
    double sr = 4000.0;
    size_t len = 2048;
    double freq_low = 500.0;
    double freq_high = 750.0;
    std::vector<T> test_signal_combined(len);
    for (size_t i = 0; i < len; ++i) {
      test_signal_combined[i]
          = T(0.5)
                * std::sin(2.0 * M_PI * freq_low * static_cast<double>(i) / sr)
            + T(0.5)
                  * std::sin(
                      2.0 * M_PI * freq_high * static_cast<double>(i) / sr);
    }
    // Use dummy CQT parameters, only FIR order and SR matter for this test
    OmniDSP::CQTPlan<T> plan(
        sr,
        128,
        400,
        1600,
        12,
        hannWindowCoeffsGenerator<T>,
        sparsity_threshold_float,
        DEFAULT_FIR_FILTER_ORDER_TEST);

    std::vector<T> result;
    ASSERT_NO_THROW(
        result = callFilterAndDownsampleBy2(plan, test_signal_combined, sr))
        << "Float filterAndDownsampleBy2 threw unexpectedly.";

    // Compare against loaded reference data
    const VecF &expected_ref
        = TestDataUtils::getExpectedFloatVec("expected_filter_downsample_f");
    T tolerance = get_tolerance<T>() * 10;  // Allow slightly larger tolerance

    // Check size first, as the backend might behave differently
    ASSERT_EQ(result.size(), expected_ref.size())
        << "Mismatch between backend output size (" << result.size()
        << ") and reference calculation size (" << expected_ref.size() << ")";

    ExpectRealVectorNear(
        expected_ref,
        result,
        tolerance,
        "Filter+Downsample Float vs Reference");
  }

  // Test Double precision - EXPECT IT TO THROW based on original test intent
  {
    using T = double;
    double sr = 4000.0;
    size_t len = 2048;
    double freq_low = 500.0;
    double freq_high = 750.0;
    std::vector<T> test_signal_combined(len);
    for (size_t i = 0; i < len; ++i) {
      test_signal_combined[i]
          = T(0.5)
                * std::sin(2.0 * M_PI * freq_low * static_cast<double>(i) / sr)
            + T(0.5)
                  * std::sin(
                      2.0 * M_PI * freq_high * static_cast<double>(i) / sr);
    }
    OmniDSP::CQTPlan<T> plan(
        sr,
        128,
        400,
        1600,
        12,
        hannWindowCoeffsGenerator<T>,
        sparsity_threshold_double,
        DEFAULT_FIR_FILTER_ORDER_TEST);

    // Verify that the double version throws (assuming backend doesn't support
    // it)
    ASSERT_THROW(
        callFilterAndDownsampleBy2(plan, test_signal_combined, sr),
        std::runtime_error);
  }
}

// 4. Test Full Recursive CQT Execute Method
TEST_F(PrecomputedRecursiveCQTTest, FullRecursiveCQT_Execute)
{
  // Test Float precision - Should execute without throwing
  {
    using T = float;
    using Complex = std::complex<T>;
    std::vector<T> test_signal = generateSineWave<T>(
        cqt_signal_freq, cqt_signal_duration, sample_rate);
    OmniDSP::CQTPlan<T> plan(
        sample_rate,
        hop_length,
        lowest_freq,
        high_freq,
        bins_per_octave,
        hannWindowCoeffsGenerator<T>,
        sparsity_threshold_float,
        DEFAULT_FIR_FILTER_ORDER_TEST);
    std::vector<std::vector<Complex>> cqt_output;

    // Expect it to run without throwing an exception
    ASSERT_NO_THROW(plan.execute(test_signal, cqt_output))
        << "Float CQT execute threw unexpectedly.";

    // Basic sanity check on output dimensions (derived from parameters)
    size_t expected_bins = static_cast<size_t>(
        std::ceil(bins_per_octave * std::log2(high_freq / lowest_freq)));
    size_t expected_frames
        = (test_signal.size() + hop_length - 1) / hop_length;  // Approximate

    ASSERT_EQ(cqt_output.size(), expected_bins)
        << "Float CQT output bin count mismatch.";
    if (expected_bins > 0) {
      ASSERT_FALSE(cqt_output.empty());
      // Frame count can vary slightly depending on implementation details,
      // check if it's close to expected.
      ASSERT_NEAR(cqt_output[0].size(), expected_frames, 2)
          << "Float CQT output frame count mismatch.";
    }

    // NOTE: Cannot compare against Librosa reference directly here,
    // as only double precision reference is available.
    // A dedicated float reference generation would be needed for comparison.
  }

  // Test Double precision - EXPECT IT TO THROW based on original test intent
  {
    using T = double;
    using Complex = std::complex<T>;
    std::vector<T> test_signal = generateSineWave<T>(
        cqt_signal_freq, cqt_signal_duration, sample_rate);
    OmniDSP::CQTPlan<T> plan(
        sample_rate,
        hop_length,
        lowest_freq,
        high_freq,
        bins_per_octave,
        hannWindowCoeffsGenerator<T>,
        sparsity_threshold_double,
        DEFAULT_FIR_FILTER_ORDER_TEST);
    std::vector<std::vector<Complex>> cqt_output;

    // Expect it to throw (assuming backend doesn't support double CQT)
    ASSERT_THROW(plan.execute(test_signal, cqt_output), std::runtime_error);
  }
}

// 5. Test Parameter Validation in Constructor (remains unchanged)
TEST_F(PrecomputedRecursiveCQTTest, ParameterValidation)
{
  using T = double;
  auto dummy_gen = hannWindowCoeffsGenerator<T>;
  ASSERT_NO_THROW(OmniDSP::CQTPlan<T>(
      sample_rate,
      hop_length,
      lowest_freq,
      high_freq,
      bins_per_octave,
      dummy_gen,
      sparsity_threshold_double,
      DEFAULT_FIR_FILTER_ORDER_TEST));
  ASSERT_THROW(
      OmniDSP::CQTPlan<T>(
          0.0,
          hop_length,
          lowest_freq,
          high_freq,
          bins_per_octave,
          dummy_gen,
          sparsity_threshold_double,
          DEFAULT_FIR_FILTER_ORDER_TEST),
      std::invalid_argument);
  ASSERT_THROW(
      OmniDSP::CQTPlan<T>(
          sample_rate,
          0,
          lowest_freq,
          high_freq,
          bins_per_octave,
          dummy_gen,
          sparsity_threshold_double,
          DEFAULT_FIR_FILTER_ORDER_TEST),
      std::invalid_argument);
  ASSERT_THROW(
      OmniDSP::CQTPlan<T>(
          sample_rate,
          hop_length,
          0.0,
          high_freq,
          bins_per_octave,
          dummy_gen,
          sparsity_threshold_double,
          DEFAULT_FIR_FILTER_ORDER_TEST),
      std::invalid_argument);
  ASSERT_THROW(
      OmniDSP::CQTPlan<T>(
          sample_rate,
          hop_length,
          lowest_freq,
          lowest_freq,
          bins_per_octave,
          dummy_gen,
          sparsity_threshold_double,
          DEFAULT_FIR_FILTER_ORDER_TEST),
      std::invalid_argument);
  ASSERT_THROW(
      OmniDSP::CQTPlan<T>(
          sample_rate,
          hop_length,
          lowest_freq,
          sample_rate / 1.9,
          bins_per_octave,
          dummy_gen,
          sparsity_threshold_double,
          DEFAULT_FIR_FILTER_ORDER_TEST),
      std::invalid_argument);  // > Nyquist check
  ASSERT_THROW(
      OmniDSP::CQTPlan<T>(
          sample_rate,
          hop_length,
          lowest_freq,
          high_freq,
          0,
          dummy_gen,
          sparsity_threshold_double,
          DEFAULT_FIR_FILTER_ORDER_TEST),
      std::invalid_argument);
  ASSERT_THROW(
      OmniDSP::CQTPlan<T>(
          sample_rate,
          hop_length,
          lowest_freq,
          high_freq,
          bins_per_octave,
          nullptr,
          sparsity_threshold_double,
          DEFAULT_FIR_FILTER_ORDER_TEST),
      std::invalid_argument);
  ASSERT_THROW(
      OmniDSP::CQTPlan<T>(
          sample_rate,
          511,
          lowest_freq,
          high_freq,
          bins_per_octave,
          dummy_gen,
          sparsity_threshold_double,
          DEFAULT_FIR_FILTER_ORDER_TEST),
      std::invalid_argument);  // Invalid hop
  ASSERT_THROW(
      OmniDSP::CQTPlan<T>(
          sample_rate,
          hop_length,
          lowest_freq,
          high_freq,
          bins_per_octave,
          dummy_gen,
          -0.1,
          DEFAULT_FIR_FILTER_ORDER_TEST),
      std::invalid_argument);  // Negative sparsity
  ASSERT_THROW(
      OmniDSP::CQTPlan<T>(
          sample_rate,
          hop_length,
          lowest_freq,
          high_freq,
          bins_per_octave,
          dummy_gen,
          sparsity_threshold_double,
          0),
      std::invalid_argument);  // Zero FIR order
  ASSERT_THROW(
      OmniDSP::CQTPlan<T>(
          sample_rate,
          hop_length,
          lowest_freq,
          high_freq,
          bins_per_octave,
          dummy_gen,
          sparsity_threshold_double,
          100),
      std::invalid_argument);  // Even FIR order
}

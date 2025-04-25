/**
 * @file tests/cpp/tests/fft.cpp
 * @brief Unit tests for the OmniDSP FFTPlan, RFFTPlan, and convenience
 * functions. Uses TestDataLoader for loading reference data and tests the
 * public API.
 */
#include "OmniDSP/fft.h"  // Include the OmniDSP FFT header

#include <cmath>    // For std::abs
#include <complex>  // For std::complex
#include <limits>   // For std::numeric_limits
#include <string>   // For std::string
#include <vector>   // For std::vector

#include "../TestDataLoader.h"  // Include the new data loader utility
#include "OmniDSP/core_types.h"  // Include for Precision, Direction, Domain enums
#include "gtest/gtest.h"         // Google Test framework

// Helper function to compare complex vectors with tolerance
template <typename T>
void ExpectComplexVectorNear(
    const std::vector<std::complex<T>> &actual,
    const std::vector<std::complex<T>> &expected,
    T abs_error,
    const std::string &message = "")
{
  ASSERT_EQ(actual.size(), expected.size())
      << message << " - Vector sizes differ. Expected: " << expected.size()
      << ", Got: " << actual.size();
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i].real(), expected[i].real(), abs_error)
        << message << " - Mismatch at index " << i << " (real part)";
    EXPECT_NEAR(actual[i].imag(), expected[i].imag(), abs_error)
        << message << " - Mismatch at index " << i << " (imaginary part)";
  }
}

// Helper function to compare real vectors with tolerance
template <typename T>
void ExpectRealVectorNear(
    const std::vector<T> &actual,
    const std::vector<T> &expected,
    T abs_error,
    const std::string &message = "")
{
  ASSERT_EQ(actual.size(), expected.size())
      << message << " - Vector sizes differ. Expected: " << expected.size()
      << ", Got: " << actual.size();
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], abs_error)
        << message << " - Mismatch at index " << i;
  }
}

// Define the test fixture
class FFT_Test : public ::testing::Test {
 protected:
  // Define constants for test parameters if needed
  const std::string suite_name = "fft";  // Corresponds to data/fft/ directory

  // Tolerance for floating point comparisons
  const double abs_error_d = 1e-9;
  const float abs_error_f = 1e-5f;

  // Helper to get tolerance based on type
  template <typename T>
  T get_tolerance()
  {
    return std::is_same_v<T, float> ? abs_error_f : abs_error_d;
  }
};

// --- FFT Tests (Complex to Complex) ---

TEST_F(FFT_Test, Plan_FFT_Forward_Double)
{
  using T = double;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Plan_FFT_Forward_Double";
  std::vector<ComplexT> input_cd, expected_cd;

  // Load input and expected data
  ASSERT_NO_THROW(
      input_cd = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cd.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_cd = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_expected_cd.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cd.empty());
  ASSERT_FALSE(expected_cd.empty());

  // Prepare output vector
  std::vector<ComplexT> output_cd(input_cd.size());

  // Create FFT plan
  OmniDSP::FFTPlan<T> plan(
      input_cd.size(),
      OmniDSP::Precision::Double,
      OmniDSP::Direction::Forward,  // Plan is for Forward direction
      OmniDSP::Domain::Complex,
      OmniDSP::FFTNorm::Forward);  // Norm used for test data generation

  // Execute FFT
  ASSERT_NO_THROW(plan.fft(input_cd, output_cd))
      << "plan.fft threw unexpectedly";

  // Compare results
  ExpectComplexVectorNear(
      output_cd, expected_cd, get_tolerance<T>(), test_case_name);
}

TEST_F(FFT_Test, Plan_FFT_Forward_Float)
{
  using T = float;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Plan_FFT_Forward_Float";
  std::vector<ComplexT> input_cf, expected_cf;

  ASSERT_NO_THROW(
      input_cf = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cf.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_cf = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_expected_cf.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cf.empty());
  ASSERT_FALSE(expected_cf.empty());

  std::vector<ComplexT> output_cf(input_cf.size());

  OmniDSP::FFTPlan<T> plan(
      input_cf.size(),
      OmniDSP::Precision::Single,
      OmniDSP::Direction::Forward,
      OmniDSP::Domain::Complex,
      OmniDSP::FFTNorm::Forward);

  ASSERT_NO_THROW(plan.fft(input_cf, output_cf))
      << "plan.fft threw unexpectedly";
  ExpectComplexVectorNear(
      output_cf, expected_cf, get_tolerance<T>(), test_case_name);
}

TEST_F(FFT_Test, Plan_FFT_Backward_Double)
{
  using T = double;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Plan_FFT_Backward_Double";
  std::vector<ComplexT> input_cd, expected_cd;

  ASSERT_NO_THROW(
      input_cd = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cd.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_cd = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_expected_cd.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cd.empty());
  ASSERT_FALSE(expected_cd.empty());

  std::vector<ComplexT> output_cd(input_cd.size());

  OmniDSP::FFTPlan<T> plan(
      input_cd.size(),
      OmniDSP::Precision::Double,
      OmniDSP::Direction::Forward,
      OmniDSP::Domain::Complex,
      OmniDSP::FFTNorm::Backward);  // Use Backward norm

  ASSERT_NO_THROW(plan.fft(input_cd, output_cd))
      << "plan.fft threw unexpectedly";
  ExpectComplexVectorNear(
      output_cd, expected_cd, get_tolerance<T>(), test_case_name);
}

TEST_F(FFT_Test, Plan_FFT_Backward_Float)
{
  using T = float;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Plan_FFT_Backward_Float";
  std::vector<ComplexT> input_cf, expected_cf;

  ASSERT_NO_THROW(
      input_cf = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cf.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_cf = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_expected_cf.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cf.empty());
  ASSERT_FALSE(expected_cf.empty());

  std::vector<ComplexT> output_cf(input_cf.size());

  OmniDSP::FFTPlan<T> plan(
      input_cf.size(),
      OmniDSP::Precision::Single,
      OmniDSP::Direction::Forward,
      OmniDSP::Domain::Complex,
      OmniDSP::FFTNorm::Backward);

  ASSERT_NO_THROW(plan.fft(input_cf, output_cf))
      << "plan.fft threw unexpectedly";
  ExpectComplexVectorNear(
      output_cf, expected_cf, get_tolerance<T>(), test_case_name);
}

TEST_F(FFT_Test, Plan_FFT_Ortho_Double)
{
  using T = double;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Plan_FFT_Ortho_Double";
  std::vector<ComplexT> input_cd, expected_cd;

  ASSERT_NO_THROW(
      input_cd = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cd.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_cd = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_expected_cd.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cd.empty());
  ASSERT_FALSE(expected_cd.empty());

  std::vector<ComplexT> output_cd(input_cd.size());

  OmniDSP::FFTPlan<T> plan(
      input_cd.size(),
      OmniDSP::Precision::Double,
      OmniDSP::Direction::Forward,
      OmniDSP::Domain::Complex,
      OmniDSP::FFTNorm::Ortho);  // Use Ortho norm

  ASSERT_NO_THROW(plan.fft(input_cd, output_cd))
      << "plan.fft threw unexpectedly";
  ExpectComplexVectorNear(
      output_cd, expected_cd, get_tolerance<T>(), test_case_name);
}

TEST_F(FFT_Test, Plan_FFT_Ortho_Float)
{
  using T = float;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Plan_FFT_Ortho_Float";
  std::vector<ComplexT> input_cf, expected_cf;

  ASSERT_NO_THROW(
      input_cf = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cf.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_cf = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_expected_cf.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cf.empty());
  ASSERT_FALSE(expected_cf.empty());

  std::vector<ComplexT> output_cf(input_cf.size());

  OmniDSP::FFTPlan<T> plan(
      input_cf.size(),
      OmniDSP::Precision::Single,
      OmniDSP::Direction::Forward,
      OmniDSP::Domain::Complex,
      OmniDSP::FFTNorm::Ortho);

  ASSERT_NO_THROW(plan.fft(input_cf, output_cf))
      << "plan.fft threw unexpectedly";
  ExpectComplexVectorNear(
      output_cf, expected_cf, get_tolerance<T>(), test_case_name);
}

// --- IFFT Tests (Complex to Complex) ---

TEST_F(FFT_Test, Plan_IFFT_Forward_Double)
{
  using T = double;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Plan_IFFT_Forward_Double";
  std::vector<ComplexT> input_cd, expected_cd;

  ASSERT_NO_THROW(
      input_cd = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cd.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_cd = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_expected_cd.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cd.empty());
  ASSERT_FALSE(expected_cd.empty());

  std::vector<ComplexT> output_cd(input_cd.size());

  // Create an Inverse plan
  OmniDSP::FFTPlan<T> plan(
      input_cd.size(),
      OmniDSP::Precision::Double,
      OmniDSP::Direction::Inverse,  // Plan is for Inverse direction
      OmniDSP::Domain::Complex,
      OmniDSP::FFTNorm::Forward);  // Match test data norm

  ASSERT_NO_THROW(plan.ifft(input_cd, output_cd))
      << "plan.ifft threw unexpectedly";
  ExpectComplexVectorNear(
      output_cd, expected_cd, get_tolerance<T>(), test_case_name);
}

TEST_F(FFT_Test, Plan_IFFT_Forward_Float)
{
  using T = float;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Plan_IFFT_Forward_Float";
  std::vector<ComplexT> input_cf, expected_cf;

  ASSERT_NO_THROW(
      input_cf = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cf.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_cf = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_expected_cf.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cf.empty());
  ASSERT_FALSE(expected_cf.empty());

  std::vector<ComplexT> output_cf(input_cf.size());

  OmniDSP::FFTPlan<T> plan(
      input_cf.size(),
      OmniDSP::Precision::Single,
      OmniDSP::Direction::Inverse,
      OmniDSP::Domain::Complex,
      OmniDSP::FFTNorm::Forward);

  ASSERT_NO_THROW(plan.ifft(input_cf, output_cf))
      << "plan.ifft threw unexpectedly";
  ExpectComplexVectorNear(
      output_cf, expected_cf, get_tolerance<T>(), test_case_name);
}

TEST_F(FFT_Test, Plan_IFFT_Backward_Double)
{
  using T = double;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Plan_IFFT_Backward_Double";
  std::vector<ComplexT> input_cd, expected_cd;

  ASSERT_NO_THROW(
      input_cd = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cd.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_cd = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_expected_cd.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cd.empty());
  ASSERT_FALSE(expected_cd.empty());

  std::vector<ComplexT> output_cd(input_cd.size());

  OmniDSP::FFTPlan<T> plan(
      input_cd.size(),
      OmniDSP::Precision::Double,
      OmniDSP::Direction::Inverse,
      OmniDSP::Domain::Complex,
      OmniDSP::FFTNorm::Backward);

  ASSERT_NO_THROW(plan.ifft(input_cd, output_cd))
      << "plan.ifft threw unexpectedly";
  ExpectComplexVectorNear(
      output_cd, expected_cd, get_tolerance<T>(), test_case_name);
}

TEST_F(FFT_Test, Plan_IFFT_Backward_Float)
{
  using T = float;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Plan_IFFT_Backward_Float";
  std::vector<ComplexT> input_cf, expected_cf;

  ASSERT_NO_THROW(
      input_cf = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cf.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_cf = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_expected_cf.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cf.empty());
  ASSERT_FALSE(expected_cf.empty());

  std::vector<ComplexT> output_cf(input_cf.size());

  OmniDSP::FFTPlan<T> plan(
      input_cf.size(),
      OmniDSP::Precision::Single,
      OmniDSP::Direction::Inverse,
      OmniDSP::Domain::Complex,
      OmniDSP::FFTNorm::Backward);

  ASSERT_NO_THROW(plan.ifft(input_cf, output_cf))
      << "plan.ifft threw unexpectedly";
  ExpectComplexVectorNear(
      output_cf, expected_cf, get_tolerance<T>(), test_case_name);
}

TEST_F(FFT_Test, Plan_IFFT_Ortho_Double)
{
  using T = double;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Plan_IFFT_Ortho_Double";
  std::vector<ComplexT> input_cd, expected_cd;

  ASSERT_NO_THROW(
      input_cd = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cd.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_cd = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_expected_cd.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cd.empty());
  ASSERT_FALSE(expected_cd.empty());

  std::vector<ComplexT> output_cd(input_cd.size());

  OmniDSP::FFTPlan<T> plan(
      input_cd.size(),
      OmniDSP::Precision::Double,
      OmniDSP::Direction::Inverse,
      OmniDSP::Domain::Complex,
      OmniDSP::FFTNorm::Ortho);

  ASSERT_NO_THROW(plan.ifft(input_cd, output_cd))
      << "plan.ifft threw unexpectedly";
  ExpectComplexVectorNear(
      output_cd, expected_cd, get_tolerance<T>(), test_case_name);
}

TEST_F(FFT_Test, Plan_IFFT_Ortho_Float)
{
  using T = float;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Plan_IFFT_Ortho_Float";
  std::vector<ComplexT> input_cf, expected_cf;

  ASSERT_NO_THROW(
      input_cf = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cf.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_cf = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_expected_cf.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cf.empty());
  ASSERT_FALSE(expected_cf.empty());

  std::vector<ComplexT> output_cf(input_cf.size());

  OmniDSP::FFTPlan<T> plan(
      input_cf.size(),
      OmniDSP::Precision::Single,
      OmniDSP::Direction::Inverse,
      OmniDSP::Domain::Complex,
      OmniDSP::FFTNorm::Ortho);

  ASSERT_NO_THROW(plan.ifft(input_cf, output_cf))
      << "plan.ifft threw unexpectedly";
  ExpectComplexVectorNear(
      output_cf, expected_cf, get_tolerance<T>(), test_case_name);
}

// --- RFFT Tests (Real to Complex) ---

TEST_F(FFT_Test, Plan_RFFT_Forward_Double)
{
  using T = double;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Plan_RFFT_Forward_Double";
  std::vector<T> input_d;
  std::vector<ComplexT> expected_cd;

  ASSERT_NO_THROW(
      input_d = TestDataLoader::loadVectorData<T>(
          suite_name, test_case_name + "_input_d.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_cd = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_expected_cd.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_d.empty());
  ASSERT_FALSE(expected_cd.empty());

  // Output size for RFFT is N/2 + 1
  size_t output_size = input_d.size() / 2 + 1;
  ASSERT_EQ(expected_cd.size(), output_size)
      << "Expected data size mismatch for RFFT";
  std::vector<ComplexT> output_cd(output_size);

  // Create RFFT plan (uses real input size N)
  OmniDSP::RFFTPlan<T> plan(
      input_d.size(),
      OmniDSP::Precision::Double,
      OmniDSP::Direction::Forward,  // Plan direction Forward
      OmniDSP::Domain::Real,
      OmniDSP::FFTNorm::Forward);  // Match test data norm

  ASSERT_NO_THROW(plan.rfft(input_d, output_cd))
      << "plan.rfft threw unexpectedly";
  ExpectComplexVectorNear(
      output_cd, expected_cd, get_tolerance<T>(), test_case_name);
}

TEST_F(FFT_Test, Plan_RFFT_Forward_Float)
{
  using T = float;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Plan_RFFT_Forward_Float";
  std::vector<T> input_f;
  std::vector<ComplexT> expected_cf;

  ASSERT_NO_THROW(
      input_f = TestDataLoader::loadVectorData<T>(
          suite_name, test_case_name + "_input_f.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_cf = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_expected_cf.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_f.empty());
  ASSERT_FALSE(expected_cf.empty());

  size_t output_size = input_f.size() / 2 + 1;
  ASSERT_EQ(expected_cf.size(), output_size)
      << "Expected data size mismatch for RFFT";
  std::vector<ComplexT> output_cf(output_size);

  OmniDSP::RFFTPlan<T> plan(
      input_f.size(),
      OmniDSP::Precision::Single,
      OmniDSP::Direction::Forward,
      OmniDSP::Domain::Real,
      OmniDSP::FFTNorm::Forward);

  ASSERT_NO_THROW(plan.rfft(input_f, output_cf))
      << "plan.rfft threw unexpectedly";
  ExpectComplexVectorNear(
      output_cf, expected_cf, get_tolerance<T>(), test_case_name);
}

// --- IRFFT Tests (Complex to Real) ---

TEST_F(FFT_Test, Plan_IRFFT_Forward_Double)
{
  using T = double;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Plan_IRFFT_Forward_Double";
  std::vector<ComplexT> input_cd;
  std::vector<T> expected_d;

  ASSERT_NO_THROW(
      input_cd = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cd.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_d = TestDataLoader::loadVectorData<T>(
          suite_name, test_case_name + "_expected_d.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cd.empty());
  ASSERT_FALSE(expected_d.empty());

  // Determine original signal length N from expected output size
  size_t n = expected_d.size();
  ASSERT_EQ(input_cd.size(), n / 2 + 1) << "Input size mismatch for IRFFT";
  std::vector<T> output_d(n);

  // Create RFFT plan (uses real output size N)
  OmniDSP::RFFTPlan<T> plan(
      n,
      OmniDSP::Precision::Double,
      OmniDSP::Direction::Inverse,  // Plan direction Inverse
      OmniDSP::Domain::Real,
      OmniDSP::FFTNorm::Forward);  // Match test data norm

  ASSERT_NO_THROW(plan.irfft(input_cd, output_d))
      << "plan.irfft threw unexpectedly";
  ExpectRealVectorNear(
      output_d, expected_d, get_tolerance<T>(), test_case_name);
}

TEST_F(FFT_Test, Plan_IRFFT_Forward_Float)
{
  using T = float;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Plan_IRFFT_Forward_Float";
  std::vector<ComplexT> input_cf;
  std::vector<T> expected_f;

  ASSERT_NO_THROW(
      input_cf = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cf.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_f = TestDataLoader::loadVectorData<T>(
          suite_name, test_case_name + "_expected_f.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cf.empty());
  ASSERT_FALSE(expected_f.empty());

  size_t n = expected_f.size();
  ASSERT_EQ(input_cf.size(), n / 2 + 1) << "Input size mismatch for IRFFT";
  std::vector<T> output_f(n);

  OmniDSP::RFFTPlan<T> plan(
      n,
      OmniDSP::Precision::Single,
      OmniDSP::Direction::Inverse,
      OmniDSP::Domain::Real,
      OmniDSP::FFTNorm::Forward);

  ASSERT_NO_THROW(plan.irfft(input_cf, output_f))
      << "plan.irfft threw unexpectedly";
  ExpectRealVectorNear(
      output_f, expected_f, get_tolerance<T>(), test_case_name);
}

TEST_F(FFT_Test, Plan_IRFFT_Backward_Double)
{
  using T = double;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Plan_IRFFT_Backward_Double";
  std::vector<ComplexT> input_cd;
  std::vector<T> expected_d;

  ASSERT_NO_THROW(
      input_cd = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cd.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_d = TestDataLoader::loadVectorData<T>(
          suite_name, test_case_name + "_expected_d.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cd.empty());
  ASSERT_FALSE(expected_d.empty());

  size_t n = expected_d.size();
  ASSERT_EQ(input_cd.size(), n / 2 + 1) << "Input size mismatch for IRFFT";
  std::vector<T> output_d(n);

  OmniDSP::RFFTPlan<T> plan(
      n,
      OmniDSP::Precision::Double,
      OmniDSP::Direction::Inverse,
      OmniDSP::Domain::Real,
      OmniDSP::FFTNorm::Backward);

  ASSERT_NO_THROW(plan.irfft(input_cd, output_d))
      << "plan.irfft threw unexpectedly";
  ExpectRealVectorNear(
      output_d, expected_d, get_tolerance<T>(), test_case_name);
}

TEST_F(FFT_Test, Plan_IRFFT_Backward_Float)
{
  using T = float;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Plan_IRFFT_Backward_Float";
  std::vector<ComplexT> input_cf;
  std::vector<T> expected_f;

  ASSERT_NO_THROW(
      input_cf = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cf.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_f = TestDataLoader::loadVectorData<T>(
          suite_name, test_case_name + "_expected_f.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cf.empty());
  ASSERT_FALSE(expected_f.empty());

  size_t n = expected_f.size();
  ASSERT_EQ(input_cf.size(), n / 2 + 1) << "Input size mismatch for IRFFT";
  std::vector<T> output_f(n);

  OmniDSP::RFFTPlan<T> plan(
      n,
      OmniDSP::Precision::Single,
      OmniDSP::Direction::Inverse,
      OmniDSP::Domain::Real,
      OmniDSP::FFTNorm::Backward);

  ASSERT_NO_THROW(plan.irfft(input_cf, output_f))
      << "plan.irfft threw unexpectedly";
  ExpectRealVectorNear(
      output_f, expected_f, get_tolerance<T>(), test_case_name);
}

TEST_F(FFT_Test, Plan_IRFFT_Ortho_Double)
{
  using T = double;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Plan_IRFFT_Ortho_Double";
  std::vector<ComplexT> input_cd;
  std::vector<T> expected_d;

  ASSERT_NO_THROW(
      input_cd = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cd.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_d = TestDataLoader::loadVectorData<T>(
          suite_name, test_case_name + "_expected_d.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cd.empty());
  ASSERT_FALSE(expected_d.empty());

  size_t n = expected_d.size();
  ASSERT_EQ(input_cd.size(), n / 2 + 1) << "Input size mismatch for IRFFT";
  std::vector<T> output_d(n);

  OmniDSP::RFFTPlan<T> plan(
      n,
      OmniDSP::Precision::Double,
      OmniDSP::Direction::Inverse,
      OmniDSP::Domain::Real,
      OmniDSP::FFTNorm::Ortho);

  ASSERT_NO_THROW(plan.irfft(input_cd, output_d))
      << "plan.irfft threw unexpectedly";
  ExpectRealVectorNear(
      output_d, expected_d, get_tolerance<T>(), test_case_name);
}

TEST_F(FFT_Test, Plan_IRFFT_Ortho_Float)
{
  using T = float;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Plan_IRFFT_Ortho_Float";
  std::vector<ComplexT> input_cf;
  std::vector<T> expected_f;

  ASSERT_NO_THROW(
      input_cf = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cf.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_f = TestDataLoader::loadVectorData<T>(
          suite_name, test_case_name + "_expected_f.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cf.empty());
  ASSERT_FALSE(expected_f.empty());

  size_t n = expected_f.size();
  ASSERT_EQ(input_cf.size(), n / 2 + 1) << "Input size mismatch for IRFFT";
  std::vector<T> output_f(n);

  OmniDSP::RFFTPlan<T> plan(
      n,
      OmniDSP::Precision::Single,
      OmniDSP::Direction::Inverse,
      OmniDSP::Domain::Real,
      OmniDSP::FFTNorm::Ortho);

  ASSERT_NO_THROW(plan.irfft(input_cf, output_f))
      << "plan.irfft threw unexpectedly";
  ExpectRealVectorNear(
      output_f, expected_f, get_tolerance<T>(), test_case_name);
}

// --- Convenience Function Tests ---

TEST_F(FFT_Test, Convenience_FFT_Double)
{
  using T = double;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Convenience_FFT_Double";
  std::vector<ComplexT> input_cd, expected_cd;

  ASSERT_NO_THROW(
      input_cd = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cd.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_cd = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_expected_cd.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cd.empty());
  ASSERT_FALSE(expected_cd.empty());

  std::vector<ComplexT> output_cd;  // Output vector will be resized by function

  // Call convenience function (default norm is Forward)
  ASSERT_NO_THROW(OmniDSP::fft(input_cd, output_cd))
      << "OmniDSP::fft threw unexpectedly";

  ExpectComplexVectorNear(
      output_cd, expected_cd, get_tolerance<T>(), test_case_name);
}

TEST_F(FFT_Test, Convenience_FFT_Float)
{
  using T = float;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Convenience_FFT_Float";
  std::vector<ComplexT> input_cf, expected_cf;

  ASSERT_NO_THROW(
      input_cf = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cf.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_cf = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_expected_cf.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cf.empty());
  ASSERT_FALSE(expected_cf.empty());

  std::vector<ComplexT> output_cf;

  ASSERT_NO_THROW(OmniDSP::fft(input_cf, output_cf))
      << "OmniDSP::fft threw unexpectedly";
  ExpectComplexVectorNear(
      output_cf, expected_cf, get_tolerance<T>(), test_case_name);
}

TEST_F(FFT_Test, Convenience_IFFT_Double)
{
  using T = double;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Convenience_IFFT_Double";
  std::vector<ComplexT> input_cd, expected_cd;

  ASSERT_NO_THROW(
      input_cd = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cd.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_cd = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_expected_cd.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cd.empty());
  ASSERT_FALSE(expected_cd.empty());

  std::vector<ComplexT> output_cd;

  // Call convenience function (default norm is Backward)
  ASSERT_NO_THROW(OmniDSP::ifft(input_cd, output_cd))
      << "OmniDSP::ifft threw unexpectedly";
  ExpectComplexVectorNear(
      output_cd, expected_cd, get_tolerance<T>(), test_case_name);
}

TEST_F(FFT_Test, Convenience_IFFT_Float)
{
  using T = float;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Convenience_IFFT_Float";
  std::vector<ComplexT> input_cf, expected_cf;

  ASSERT_NO_THROW(
      input_cf = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cf.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_cf = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_expected_cf.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cf.empty());
  ASSERT_FALSE(expected_cf.empty());

  std::vector<ComplexT> output_cf;

  ASSERT_NO_THROW(OmniDSP::ifft(input_cf, output_cf))
      << "OmniDSP::ifft threw unexpectedly";
  ExpectComplexVectorNear(
      output_cf, expected_cf, get_tolerance<T>(), test_case_name);
}

TEST_F(FFT_Test, Convenience_RFFT_Double)
{
  using T = double;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Convenience_RFFT_Double";
  std::vector<T> input_d;
  std::vector<ComplexT> expected_cd;

  ASSERT_NO_THROW(
      input_d = TestDataLoader::loadVectorData<T>(
          suite_name, test_case_name + "_input_d.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_cd = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_expected_cd.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_d.empty());
  ASSERT_FALSE(expected_cd.empty());

  std::vector<ComplexT> output_cd;

  // Call convenience function (default norm is Forward)
  ASSERT_NO_THROW(OmniDSP::rfft(input_d, output_cd))
      << "OmniDSP::rfft threw unexpectedly";
  ExpectComplexVectorNear(
      output_cd, expected_cd, get_tolerance<T>(), test_case_name);
}

TEST_F(FFT_Test, Convenience_RFFT_Float)
{
  using T = float;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Convenience_RFFT_Float";
  std::vector<T> input_f;
  std::vector<ComplexT> expected_cf;

  ASSERT_NO_THROW(
      input_f = TestDataLoader::loadVectorData<T>(
          suite_name, test_case_name + "_input_f.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_cf = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_expected_cf.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_f.empty());
  ASSERT_FALSE(expected_cf.empty());

  std::vector<ComplexT> output_cf;

  ASSERT_NO_THROW(OmniDSP::rfft(input_f, output_cf))
      << "OmniDSP::rfft threw unexpectedly";
  ExpectComplexVectorNear(
      output_cf, expected_cf, get_tolerance<T>(), test_case_name);
}

TEST_F(FFT_Test, Convenience_IRFFT_Double)
{
  using T = double;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Convenience_IRFFT_Double";
  std::vector<ComplexT> input_cd;
  std::vector<T> expected_d;

  ASSERT_NO_THROW(
      input_cd = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cd.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_d = TestDataLoader::loadVectorData<T>(
          suite_name, test_case_name + "_expected_d.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cd.empty());
  ASSERT_FALSE(expected_d.empty());

  size_t n = expected_d.size();
  ASSERT_EQ(input_cd.size(), n / 2 + 1) << "Input size mismatch for IRFFT";
  std::vector<T> output_d;

  // Call convenience function (default norm is Backward, need N)
  ASSERT_NO_THROW(OmniDSP::irfft(input_cd, output_d, n))
      << "OmniDSP::irfft threw unexpectedly";
  ExpectRealVectorNear(
      output_d, expected_d, get_tolerance<T>(), test_case_name);
}

TEST_F(FFT_Test, Convenience_IRFFT_Float)
{
  using T = float;
  using ComplexT = std::complex<T>;
  std::string test_case_name = "Convenience_IRFFT_Float";
  std::vector<ComplexT> input_cf;
  std::vector<T> expected_f;

  ASSERT_NO_THROW(
      input_cf = TestDataLoader::loadVectorData<ComplexT>(
          suite_name, test_case_name + "_input_cf.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      expected_f = TestDataLoader::loadVectorData<T>(
          suite_name, test_case_name + "_expected_f.txt"))
      << "Failed to load expected";
  ASSERT_FALSE(input_cf.empty());
  ASSERT_FALSE(expected_f.empty());

  size_t n = expected_f.size();
  ASSERT_EQ(input_cf.size(), n / 2 + 1) << "Input size mismatch for IRFFT";
  std::vector<T> output_f;

  ASSERT_NO_THROW(OmniDSP::irfft(input_cf, output_f, n))
      << "OmniDSP::irfft threw unexpectedly";
  ExpectRealVectorNear(
      output_f, expected_f, get_tolerance<T>(), test_case_name);
}

// TODO: Add tests for different norms in convenience functions if needed
// TODO: Add tests for edge cases (e.g., empty input, size 1 input)
// TODO: Add tests for the in-place convenience function overloads if
// implemented

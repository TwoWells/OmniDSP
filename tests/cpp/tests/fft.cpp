#include "OmniDSP/fft.h"  // Include the OmniDSP FFT header

#include <cmath>    // For std::abs
#include <complex>  // For std::complex
#include <limits>   // For std::numeric_limits
#include <string>   // For std::string
#include <vector>   // For std::vector

#include "../TestDataLoader.h"  // Include the new data loader utility
#include "gtest/gtest.h"        // Google Test framework

// Helper function to compare complex vectors with tolerance
template <typename T>
void ExpectComplexVectorNear(const std::vector<std::complex<T>> &actual,
                             const std::vector<std::complex<T>> &expected,
                             T abs_error, const std::string &message = "") {
  ASSERT_EQ(actual.size(), expected.size())
      << message << " - Vector sizes differ.";
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i].real(), expected[i].real(), abs_error)
        << message << " - Mismatch at index " << i << " (real part)";
    EXPECT_NEAR(actual[i].imag(), expected[i].imag(), abs_error)
        << message << " - Mismatch at index " << i << " (imaginary part)";
  }
}

// Helper function to compare real vectors with tolerance
template <typename T>
void ExpectRealVectorNear(const std::vector<T> &actual,
                          const std::vector<T> &expected, T abs_error,
                          const std::string &message = "") {
  ASSERT_EQ(actual.size(), expected.size())
      << message << " - Vector sizes differ.";
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

  // You can add SetUp() or TearDown() methods if needed
};

// --- FFT Tests (Complex to Complex) ---

TEST_F(FFT_Test, Plan_FFT_Forward_Double) {
  std::string test_case_name = "Plan_FFT_Forward_Double";
  // Load input and expected data using the new loader
  auto input_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_input_cd.txt");
  auto expected_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_expected_cd.txt");

  // Prepare output vector
  std::vector<std::complex<double>> output_cd(input_cd.size());

  // Create FFT plan
  OmniDSP::FFTPlan<double> plan(input_cd.size());

  // Execute FFT
  plan.fft(input_cd, output_cd, OmniDSP::FFTNorm::Forward);  // Use Forward norm

  // Compare results
  ExpectComplexVectorNear(output_cd, expected_cd, abs_error_d, test_case_name);
}

TEST_F(FFT_Test, Plan_FFT_Forward_Float) {
  std::string test_case_name = "Plan_FFT_Forward_Float";
  auto input_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_input_cf.txt");
  auto expected_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_expected_cf.txt");
  std::vector<std::complex<float>> output_cf(input_cf.size());
  OmniDSP::FFTPlan<float> plan(input_cf.size());
  plan.fft(input_cf, output_cf, OmniDSP::FFTNorm::Forward);
  ExpectComplexVectorNear(output_cf, expected_cf, abs_error_f, test_case_name);
}

TEST_F(FFT_Test, Plan_FFT_Backward_Double) {
  std::string test_case_name = "Plan_FFT_Backward_Double";
  auto input_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_input_cd.txt");
  auto expected_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_expected_cd.txt");
  std::vector<std::complex<double>> output_cd(input_cd.size());
  OmniDSP::FFTPlan<double> plan(input_cd.size());
  plan.fft(input_cd, output_cd,
           OmniDSP::FFTNorm::Backward);  // Use Backward norm
  ExpectComplexVectorNear(output_cd, expected_cd, abs_error_d, test_case_name);
}

TEST_F(FFT_Test, Plan_FFT_Backward_Float) {
  std::string test_case_name = "Plan_FFT_Backward_Float";
  auto input_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_input_cf.txt");
  auto expected_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_expected_cf.txt");
  std::vector<std::complex<float>> output_cf(input_cf.size());
  OmniDSP::FFTPlan<float> plan(input_cf.size());
  plan.fft(input_cf, output_cf, OmniDSP::FFTNorm::Backward);
  ExpectComplexVectorNear(output_cf, expected_cf, abs_error_f, test_case_name);
}

TEST_F(FFT_Test, Plan_FFT_Ortho_Double) {
  std::string test_case_name = "Plan_FFT_Ortho_Double";
  auto input_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_input_cd.txt");
  auto expected_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_expected_cd.txt");
  std::vector<std::complex<double>> output_cd(input_cd.size());
  OmniDSP::FFTPlan<double> plan(input_cd.size());
  plan.fft(input_cd, output_cd, OmniDSP::FFTNorm::Ortho);  // Use Ortho norm
  ExpectComplexVectorNear(output_cd, expected_cd, abs_error_d, test_case_name);
}

TEST_F(FFT_Test, Plan_FFT_Ortho_Float) {
  std::string test_case_name = "Plan_FFT_Ortho_Float";
  auto input_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_input_cf.txt");
  auto expected_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_expected_cf.txt");
  std::vector<std::complex<float>> output_cf(input_cf.size());
  OmniDSP::FFTPlan<float> plan(input_cf.size());
  plan.fft(input_cf, output_cf, OmniDSP::FFTNorm::Ortho);
  ExpectComplexVectorNear(output_cf, expected_cf, abs_error_f, test_case_name);
}

// --- IFFT Tests (Complex to Complex) ---

TEST_F(FFT_Test, Plan_IFFT_Forward_Double) {
  std::string test_case_name = "Plan_IFFT_Forward_Double";
  auto input_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_input_cd.txt");
  auto expected_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_expected_cd.txt");
  std::vector<std::complex<double>> output_cd(input_cd.size());
  OmniDSP::FFTPlan<double> plan(input_cd.size());
  plan.ifft(input_cd, output_cd,
            OmniDSP::FFTNorm::Forward);  // Use Forward norm
  ExpectComplexVectorNear(output_cd, expected_cd, abs_error_d, test_case_name);
}

TEST_F(FFT_Test, Plan_IFFT_Forward_Float) {
  std::string test_case_name = "Plan_IFFT_Forward_Float";
  auto input_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_input_cf.txt");
  auto expected_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_expected_cf.txt");
  std::vector<std::complex<float>> output_cf(input_cf.size());
  OmniDSP::FFTPlan<float> plan(input_cf.size());
  plan.ifft(input_cf, output_cf, OmniDSP::FFTNorm::Forward);
  ExpectComplexVectorNear(output_cf, expected_cf, abs_error_f, test_case_name);
}

TEST_F(FFT_Test, Plan_IFFT_Backward_Double) {
  std::string test_case_name = "Plan_IFFT_Backward_Double";
  auto input_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_input_cd.txt");
  auto expected_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_expected_cd.txt");
  std::vector<std::complex<double>> output_cd(input_cd.size());
  OmniDSP::FFTPlan<double> plan(input_cd.size());
  plan.ifft(input_cd, output_cd,
            OmniDSP::FFTNorm::Backward);  // Use Backward norm
  ExpectComplexVectorNear(output_cd, expected_cd, abs_error_d, test_case_name);
}

TEST_F(FFT_Test, Plan_IFFT_Backward_Float) {
  std::string test_case_name = "Plan_IFFT_Backward_Float";
  auto input_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_input_cf.txt");
  auto expected_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_expected_cf.txt");
  std::vector<std::complex<float>> output_cf(input_cf.size());
  OmniDSP::FFTPlan<float> plan(input_cf.size());
  plan.ifft(input_cf, output_cf, OmniDSP::FFTNorm::Backward);
  ExpectComplexVectorNear(output_cf, expected_cf, abs_error_f, test_case_name);
}

TEST_F(FFT_Test, Plan_IFFT_Ortho_Double) {
  std::string test_case_name = "Plan_IFFT_Ortho_Double";
  auto input_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_input_cd.txt");
  auto expected_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_expected_cd.txt");
  std::vector<std::complex<double>> output_cd(input_cd.size());
  OmniDSP::FFTPlan<double> plan(input_cd.size());
  plan.ifft(input_cd, output_cd, OmniDSP::FFTNorm::Ortho);  // Use Ortho norm
  ExpectComplexVectorNear(output_cd, expected_cd, abs_error_d, test_case_name);
}

TEST_F(FFT_Test, Plan_IFFT_Ortho_Float) {
  std::string test_case_name = "Plan_IFFT_Ortho_Float";
  auto input_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_input_cf.txt");
  auto expected_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_expected_cf.txt");
  std::vector<std::complex<float>> output_cf(input_cf.size());
  OmniDSP::FFTPlan<float> plan(input_cf.size());
  plan.ifft(input_cf, output_cf, OmniDSP::FFTNorm::Ortho);
  ExpectComplexVectorNear(output_cf, expected_cf, abs_error_f, test_case_name);
}

// --- RFFT Tests (Real to Complex) ---

TEST_F(FFT_Test, Plan_RFFT_Forward_Double) {
  std::string test_case_name = "Plan_RFFT_Forward_Double";
  // Input is real double
  auto input_d = TestDataLoader::loadVectorData<double>(
      suite_name, test_case_name + "_input_d.txt");
  // Expected output is complex double
  auto expected_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_expected_cd.txt");

  // Output size for RFFT is N/2 + 1
  size_t output_size = input_d.size() / 2 + 1;
  std::vector<std::complex<double>> output_cd(output_size);

  // Create RFFT plan (uses input size N)
  OmniDSP::RFFTPlan<double> plan(input_d.size());

  // Execute RFFT
  plan.rfft(input_d, output_cd, OmniDSP::FFTNorm::Forward);  // Use Forward norm

  // Compare results
  ExpectComplexVectorNear(output_cd, expected_cd, abs_error_d, test_case_name);
}

TEST_F(FFT_Test, Plan_RFFT_Forward_Float) {
  std::string test_case_name = "Plan_RFFT_Forward_Float";
  auto input_f = TestDataLoader::loadVectorData<float>(
      suite_name, test_case_name + "_input_f.txt");
  auto expected_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_expected_cf.txt");
  size_t output_size = input_f.size() / 2 + 1;
  std::vector<std::complex<float>> output_cf(output_size);
  OmniDSP::RFFTPlan<float> plan(input_f.size());
  plan.rfft(input_f, output_cf, OmniDSP::FFTNorm::Forward);
  ExpectComplexVectorNear(output_cf, expected_cf, abs_error_f, test_case_name);
}

// Add tests for RFFT Backward and Ortho norms if needed, following the pattern

// --- IRFFT Tests (Complex to Real) ---
// Note: IRFFT input size is N/2 + 1, output size is N

TEST_F(FFT_Test, Plan_IRFFT_Forward_Double) {
  std::string test_case_name = "Plan_IRFFT_Forward_Double";
  // Input is complex double (size N/2 + 1)
  auto input_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_input_cd.txt");
  // Expected output is real double (size N)
  auto expected_d = TestDataLoader::loadVectorData<double>(
      suite_name, test_case_name + "_expected_d.txt");

  // Determine original signal length N from expected output size
  size_t n = expected_d.size();
  ASSERT_EQ(input_cd.size(), n / 2 + 1) << "Input size mismatch for IRFFT";

  std::vector<double> output_d(n);

  // Create RFFT plan (uses output size N)
  OmniDSP::RFFTPlan<double> plan(n);

  // Execute IRFFT
  plan.irfft(input_cd, output_d,
             OmniDSP::FFTNorm::Forward);  // Use Forward norm

  // Compare results
  ExpectRealVectorNear(output_d, expected_d, abs_error_d, test_case_name);
}

TEST_F(FFT_Test, Plan_IRFFT_Forward_Float) {
  std::string test_case_name = "Plan_IRFFT_Forward_Float";
  auto input_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_input_cf.txt");
  auto expected_f = TestDataLoader::loadVectorData<float>(
      suite_name, test_case_name + "_expected_f.txt");
  size_t n = expected_f.size();
  ASSERT_EQ(input_cf.size(), n / 2 + 1) << "Input size mismatch for IRFFT";
  std::vector<float> output_f(n);
  OmniDSP::RFFTPlan<float> plan(n);
  plan.irfft(input_cf, output_f, OmniDSP::FFTNorm::Forward);
  ExpectRealVectorNear(output_f, expected_f, abs_error_f, test_case_name);
}

TEST_F(FFT_Test, Plan_IRFFT_Backward_Double) {
  std::string test_case_name = "Plan_IRFFT_Backward_Double";
  auto input_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_input_cd.txt");
  auto expected_d = TestDataLoader::loadVectorData<double>(
      suite_name, test_case_name + "_expected_d.txt");
  size_t n = expected_d.size();
  ASSERT_EQ(input_cd.size(), n / 2 + 1) << "Input size mismatch for IRFFT";
  std::vector<double> output_d(n);
  OmniDSP::RFFTPlan<double> plan(n);
  plan.irfft(input_cd, output_d,
             OmniDSP::FFTNorm::Backward);  // Use Backward norm
  ExpectRealVectorNear(output_d, expected_d, abs_error_d, test_case_name);
}

TEST_F(FFT_Test, Plan_IRFFT_Backward_Float) {
  std::string test_case_name = "Plan_IRFFT_Backward_Float";
  auto input_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_input_cf.txt");
  auto expected_f = TestDataLoader::loadVectorData<float>(
      suite_name, test_case_name + "_expected_f.txt");
  size_t n = expected_f.size();
  ASSERT_EQ(input_cf.size(), n / 2 + 1) << "Input size mismatch for IRFFT";
  std::vector<float> output_f(n);
  OmniDSP::RFFTPlan<float> plan(n);
  plan.irfft(input_cf, output_f, OmniDSP::FFTNorm::Backward);
  ExpectRealVectorNear(output_f, expected_f, abs_error_f, test_case_name);
}

TEST_F(FFT_Test, Plan_IRFFT_Ortho_Double) {
  std::string test_case_name = "Plan_IRFFT_Ortho_Double";
  auto input_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_input_cd.txt");
  auto expected_d = TestDataLoader::loadVectorData<double>(
      suite_name, test_case_name + "_expected_d.txt");
  size_t n = expected_d.size();
  ASSERT_EQ(input_cd.size(), n / 2 + 1) << "Input size mismatch for IRFFT";
  std::vector<double> output_d(n);
  OmniDSP::RFFTPlan<double> plan(n);
  plan.irfft(input_cd, output_d, OmniDSP::FFTNorm::Ortho);  // Use Ortho norm
  ExpectRealVectorNear(output_d, expected_d, abs_error_d, test_case_name);
}

TEST_F(FFT_Test, Plan_IRFFT_Ortho_Float) {
  std::string test_case_name = "Plan_IRFFT_Ortho_Float";
  auto input_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_input_cf.txt");
  auto expected_f = TestDataLoader::loadVectorData<float>(
      suite_name, test_case_name + "_expected_f.txt");
  size_t n = expected_f.size();
  ASSERT_EQ(input_cf.size(), n / 2 + 1) << "Input size mismatch for IRFFT";
  std::vector<float> output_f(n);
  OmniDSP::RFFTPlan<float> plan(n);
  plan.irfft(input_cf, output_f, OmniDSP::FFTNorm::Ortho);
  ExpectRealVectorNear(output_f, expected_f, abs_error_f, test_case_name);
}

// --- Convenience Function Tests ---
// These use the non-plan based functions

TEST_F(FFT_Test, Convenience_FFT_Double) {
  std::string test_case_name =
      "Convenience_FFT_Double";  // Assuming data files exist for this
  auto input_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_input_cd.txt");
  auto expected_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_expected_cd.txt");
  std::vector<std::complex<double>> output_cd(input_cd.size());

  OmniDSP::fft(input_cd, output_cd);  // Default norm is Forward

  ExpectComplexVectorNear(output_cd, expected_cd, abs_error_d, test_case_name);
}

TEST_F(FFT_Test, Convenience_FFT_Float) {
  std::string test_case_name = "Convenience_FFT_Float";
  auto input_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_input_cf.txt");
  auto expected_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_expected_cf.txt");
  std::vector<std::complex<float>> output_cf(input_cf.size());

  OmniDSP::fft(input_cf, output_cf);

  ExpectComplexVectorNear(output_cf, expected_cf, abs_error_f, test_case_name);
}

TEST_F(FFT_Test, Convenience_IFFT_Double) {
  std::string test_case_name = "Convenience_IFFT_Double";
  auto input_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_input_cd.txt");
  auto expected_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_expected_cd.txt");
  std::vector<std::complex<double>> output_cd(input_cd.size());

  OmniDSP::ifft(input_cd, output_cd);  // Default norm is Backward

  ExpectComplexVectorNear(output_cd, expected_cd, abs_error_d, test_case_name);
}

TEST_F(FFT_Test, Convenience_IFFT_Float) {
  std::string test_case_name = "Convenience_IFFT_Float";
  auto input_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_input_cf.txt");
  auto expected_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_expected_cf.txt");
  std::vector<std::complex<float>> output_cf(input_cf.size());

  OmniDSP::ifft(input_cf, output_cf);

  ExpectComplexVectorNear(output_cf, expected_cf, abs_error_f, test_case_name);
}

TEST_F(FFT_Test, Convenience_RFFT_Double) {
  std::string test_case_name = "Convenience_RFFT_Double";
  auto input_d = TestDataLoader::loadVectorData<double>(
      suite_name, test_case_name + "_input_d.txt");
  auto expected_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_expected_cd.txt");
  size_t output_size = input_d.size() / 2 + 1;
  std::vector<std::complex<double>> output_cd(output_size);

  OmniDSP::rfft(input_d, output_cd);  // Default norm is Forward

  ExpectComplexVectorNear(output_cd, expected_cd, abs_error_d, test_case_name);
}

TEST_F(FFT_Test, Convenience_RFFT_Float) {
  std::string test_case_name = "Convenience_RFFT_Float";
  auto input_f = TestDataLoader::loadVectorData<float>(
      suite_name, test_case_name + "_input_f.txt");
  auto expected_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_expected_cf.txt");
  size_t output_size = input_f.size() / 2 + 1;
  std::vector<std::complex<float>> output_cf(output_size);

  OmniDSP::rfft(input_f, output_cf);

  ExpectComplexVectorNear(output_cf, expected_cf, abs_error_f, test_case_name);
}

TEST_F(FFT_Test, Convenience_IRFFT_Double) {
  std::string test_case_name = "Convenience_IRFFT_Double";
  auto input_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_input_cd.txt");
  auto expected_d = TestDataLoader::loadVectorData<double>(
      suite_name, test_case_name + "_expected_d.txt");
  size_t n = expected_d.size();
  ASSERT_EQ(input_cd.size(), n / 2 + 1) << "Input size mismatch for IRFFT";
  std::vector<double> output_d(n);

  OmniDSP::irfft(input_cd, output_d, n);  // Default norm is Backward, need N

  ExpectRealVectorNear(output_d, expected_d, abs_error_d, test_case_name);
}

TEST_F(FFT_Test, Convenience_IRFFT_Float) {
  std::string test_case_name = "Convenience_IRFFT_Float";
  auto input_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_input_cf.txt");
  auto expected_f = TestDataLoader::loadVectorData<float>(
      suite_name, test_case_name + "_expected_f.txt");
  size_t n = expected_f.size();
  ASSERT_EQ(input_cf.size(), n / 2 + 1) << "Input size mismatch for IRFFT";
  std::vector<float> output_f(n);

  OmniDSP::irfft(input_cf, output_f, n);

  ExpectRealVectorNear(output_f, expected_f, abs_error_f, test_case_name);
}

// TODO: Add tests for different norms in convenience functions if needed
// TODO: Add tests for edge cases (e.g., empty input, size 1 input)

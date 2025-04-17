/**
 * @file tests/cpp/window.cpp
 * @brief Unit tests for the OmniDSP Window class.
 * Corrected include path. Added explicit cast for double->float conversion.
 */
#include <OmniDSP/window.h>  // Use the renamed window header
#include <gtest/gtest.h>

#include <algorithm>  // For std::copy, std::transform
#include <cmath>
#include <limits>   // For std::numeric_limits
#include <numeric>  // For std::accumulate
#include <vector>

// Define M_PI if it's not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper to compare vectors
template <typename T>
void ExpectVectorNear(const std::vector<T>& expected,
                      const std::vector<T>& actual, T tolerance) {
  ASSERT_EQ(expected.size(), actual.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(expected[i], actual[i], tolerance) << "at index " << i;
  }
}

// Test fixture for Window tests
class WindowTest : public ::testing::Test {
 protected:
  // Define test signal
  std::vector<float> signal_f = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<double> signal_d = {1.0, 2.0, 3.0, 4.0, 5.0};
  size_t N = 5;

  // Tolerance for floating point comparisons
  float tol_f = std::numeric_limits<float>::epsilon() * 100;
  double tol_d = std::numeric_limits<double>::epsilon() * 1000;

  // Manually calculated expected results for N=5
  // Hann: w[n] = 0.5 - 0.5 * cos(2*pi*n / (N-1))
  std::vector<double> hann_coeffs_d = {
      0.5 - 0.5 * cos(2.0 * M_PI * 0 / 4.0),  // 0.0
      0.5 - 0.5 * cos(2.0 * M_PI * 1 / 4.0),  // 0.5
      0.5 - 0.5 * cos(2.0 * M_PI * 2 / 4.0),  // 1.0
      0.5 - 0.5 * cos(2.0 * M_PI * 3 / 4.0),  // 0.5
      0.5 - 0.5 * cos(2.0 * M_PI * 4 / 4.0)   // 0.0
  };
  std::vector<float> hann_coeffs_f = {0.0f, 0.5f, 1.0f, 0.5f, 0.0f};  // Approx

  // Hamming: w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
  std::vector<double> hamming_coeffs_d = {
      0.54 - 0.46 * cos(2.0 * M_PI * 0 / 4.0),  // 0.08
      0.54 - 0.46 * cos(2.0 * M_PI * 1 / 4.0),  // 0.54
      0.54 - 0.46 * cos(2.0 * M_PI * 2 / 4.0),  // 1.0
      0.54 - 0.46 * cos(2.0 * M_PI * 3 / 4.0),  // 0.54
      0.54 - 0.46 * cos(2.0 * M_PI * 4 / 4.0)   // 0.08
  };
  std::vector<float> hamming_coeffs_f;  // Calculate from double

  // Kaiser: Requires Bessel function, harder to precompute simply. Test
  // shape/endpoints. beta = 8.0
  double kaiser_beta = 8.0;
  double i0_beta = std::cyl_bessel_i(0.0, kaiser_beta);
  std::vector<double> kaiser_coeffs_d = {
      std::cyl_bessel_i(0.0,
                        kaiser_beta* sqrt(1.0 - pow(2.0 * 0 / 4.0 - 1.0, 2))) /
          i0_beta,
      std::cyl_bessel_i(0.0,
                        kaiser_beta* sqrt(1.0 - pow(2.0 * 1 / 4.0 - 1.0, 2))) /
          i0_beta,
      std::cyl_bessel_i(0.0,
                        kaiser_beta* sqrt(1.0 - pow(2.0 * 2 / 4.0 - 1.0, 2))) /
          i0_beta,
      std::cyl_bessel_i(0.0,
                        kaiser_beta* sqrt(1.0 - pow(2.0 * 3 / 4.0 - 1.0, 2))) /
          i0_beta,
      std::cyl_bessel_i(0.0,
                        kaiser_beta* sqrt(1.0 - pow(2.0 * 4 / 4.0 - 1.0, 2))) /
          i0_beta};
  std::vector<float> kaiser_coeffs_f;  // Calculate from double

  // Flattop: Use coefficients a0-a4
  double ft_a0 = 0.21557895, ft_a1 = 0.41663158, ft_a2 = 0.277263158,
         ft_a3 = 0.083578947, ft_a4 = 0.006947368;
  std::vector<double> flattop_coeffs_d = {
      ft_a0 - ft_a1 * cos(2 * M_PI * 0 / 4.) + ft_a2 * cos(4 * M_PI * 0 / 4.) -
          ft_a3 * cos(6 * M_PI * 0 / 4.) + ft_a4 * cos(8 * M_PI * 0 / 4.),
      ft_a0 - ft_a1* cos(2 * M_PI * 1 / 4.) + ft_a2* cos(4 * M_PI * 1 / 4.) -
          ft_a3* cos(6 * M_PI * 1 / 4.) + ft_a4* cos(8 * M_PI * 1 / 4.),
      ft_a0 - ft_a1* cos(2 * M_PI * 2 / 4.) + ft_a2* cos(4 * M_PI * 2 / 4.) -
          ft_a3* cos(6 * M_PI * 2 / 4.) + ft_a4* cos(8 * M_PI * 2 / 4.),
      ft_a0 - ft_a1* cos(2 * M_PI * 3 / 4.) + ft_a2* cos(4 * M_PI * 3 / 4.) -
          ft_a3* cos(6 * M_PI * 3 / 4.) + ft_a4* cos(8 * M_PI * 3 / 4.),
      ft_a0 - ft_a1* cos(2 * M_PI * 4 / 4.) + ft_a2* cos(4 * M_PI * 4 / 4.) -
          ft_a3* cos(6 * M_PI * 4 / 4.) + ft_a4* cos(8 * M_PI * 4 / 4.)};
  std::vector<float> flattop_coeffs_f;  // Calculate from double

  // Expected windowed signals
  std::vector<float> expected_hann_f;
  std::vector<double> expected_hann_d;
  std::vector<float> expected_hamming_f;
  std::vector<double> expected_hamming_d;
  std::vector<float> expected_kaiser_f;
  std::vector<double> expected_kaiser_d;
  std::vector<float> expected_flattop_f;
  std::vector<double> expected_flattop_d;

  // Setup to calculate expected values
  void SetUp() override {
    // Convert double coeffs to float using std::transform with static_cast
    hamming_coeffs_f.resize(N);
    std::transform(hamming_coeffs_d.begin(), hamming_coeffs_d.end(),
                   hamming_coeffs_f.begin(),
                   [](double d) { return static_cast<float>(d); });

    kaiser_coeffs_f.resize(N);
    std::transform(kaiser_coeffs_d.begin(), kaiser_coeffs_d.end(),
                   kaiser_coeffs_f.begin(),
                   [](double d) { return static_cast<float>(d); });

    flattop_coeffs_f.resize(N);
    std::transform(flattop_coeffs_d.begin(), flattop_coeffs_d.end(),
                   flattop_coeffs_f.begin(),
                   [](double d) { return static_cast<float>(d); });

    // Calculate expected windowed signals
    expected_hann_f.resize(N);
    expected_hann_d.resize(N);
    expected_hamming_f.resize(N);
    expected_hamming_d.resize(N);
    expected_kaiser_f.resize(N);
    expected_kaiser_d.resize(N);
    expected_flattop_f.resize(N);
    expected_flattop_d.resize(N);
    for (size_t i = 0; i < N; ++i) {
      expected_hann_f[i] = signal_f[i] * hann_coeffs_f[i];
      expected_hann_d[i] = signal_d[i] * hann_coeffs_d[i];
      expected_hamming_f[i] = signal_f[i] * hamming_coeffs_f[i];
      expected_hamming_d[i] = signal_d[i] * hamming_coeffs_d[i];
      expected_kaiser_f[i] = signal_f[i] * kaiser_coeffs_f[i];
      expected_kaiser_d[i] = signal_d[i] * kaiser_coeffs_d[i];
      expected_flattop_f[i] = signal_f[i] * flattop_coeffs_f[i];
      expected_flattop_d[i] = signal_d[i] * flattop_coeffs_d[i];
    }
  }
};

// Test Hann window
TEST_F(WindowTest, Hann) {
  auto result_f = OmniDSP::Window::hann(signal_f);
  ExpectVectorNear(expected_hann_f, result_f, tol_f);

  auto result_d = OmniDSP::Window::hann(signal_d);
  ExpectVectorNear(expected_hann_d, result_d, tol_d);
}

// Test Hamming window
TEST_F(WindowTest, Hamming) {
  auto result_f = OmniDSP::Window::hamming(signal_f);
  ExpectVectorNear(expected_hamming_f, result_f, tol_f);

  auto result_d = OmniDSP::Window::hamming(signal_d);
  ExpectVectorNear(expected_hamming_d, result_d, tol_d);
}

// Test Kaiser window
TEST_F(WindowTest, Kaiser) {
  auto result_f =
      OmniDSP::Window::kaiser(signal_f, static_cast<float>(kaiser_beta));
  ExpectVectorNear(expected_kaiser_f, result_f, tol_f);

  auto result_d = OmniDSP::Window::kaiser(signal_d, kaiser_beta);
  ExpectVectorNear(expected_kaiser_d, result_d, tol_d);
}

// Test Flattop window
TEST_F(WindowTest, Flattop) {
  auto result_f = OmniDSP::Window::flattop(signal_f);
  ExpectVectorNear(expected_flattop_f, result_f, tol_f);

  auto result_d = OmniDSP::Window::flattop(signal_d);
  ExpectVectorNear(expected_flattop_d, result_d, tol_d);
}

// Test edge cases
TEST_F(WindowTest, EdgeCases) {
  std::vector<float> empty_f;
  std::vector<double> empty_d;
  std::vector<float> single_f = {1.0f};
  std::vector<double> single_d = {1.0};

  // Expect throw for empty input
  EXPECT_THROW(OmniDSP::Window::hann(empty_f), std::invalid_argument);
  EXPECT_THROW(OmniDSP::Window::hamming(empty_d), std::invalid_argument);
  EXPECT_THROW(OmniDSP::Window::kaiser(empty_f, 8.0f), std::invalid_argument);
  EXPECT_THROW(OmniDSP::Window::flattop(empty_d), std::invalid_argument);

  // Expect single output of input value for N=1 (window coeff is 1)
  EXPECT_EQ(OmniDSP::Window::hann(single_f)[0], 1.0f);
  EXPECT_EQ(OmniDSP::Window::hamming(single_d)[0], 1.0);
  EXPECT_EQ(OmniDSP::Window::kaiser(single_f, 8.0f)[0],
            1.0f);  // Beta doesn't matter for N=1
  EXPECT_EQ(OmniDSP::Window::flattop(single_d)[0], 1.0);

  // Expect throw for negative beta in Kaiser
  EXPECT_THROW(OmniDSP::Window::kaiser(signal_f, -1.0f), std::invalid_argument);
}

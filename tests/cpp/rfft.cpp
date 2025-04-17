// tests/rfft.cpp
// Tests for forward Real-to-Complex FFT functions (rfft, FFTPlan::execute_rfft)

#include <OmniDSP/omnidsp.h>  // Use correct include path
#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <limits>
#include <vector>

// --- Test Fixture ---
class FFT_R2C_Forward_Test : public ::testing::Test {
 protected:
  // Using a power-of-2 size required by Accelerate REAL domain implementation
  const size_t N = 16;
  const size_t Nc = N / 2 + 1;  // Expected complex output size
  const double double_tolerance =
      std::numeric_limits<double>::epsilon() * N * 10;
  const float float_tolerance = std::numeric_limits<float>::epsilon() * N * 10;

  // Helper to compare complex vectors (copied - consider common header)
  template <typename T>
  void ExpectComplexVectorNear(const std::vector<std::complex<T>>& expected,
                               const std::vector<std::complex<T>>& actual,
                               T tolerance, const std::string& msg = "") {
    ASSERT_EQ(expected.size(), actual.size()) << msg << " - Size mismatch";
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(expected[i].real(), actual[i].real(), tolerance)
          << msg << " - Mismatch at index " << i << " (real)";
      EXPECT_NEAR(expected[i].imag(), actual[i].imag(), tolerance)
          << msg << " - Mismatch at index " << i << " (imag)";
    }
  }
};

// --- Test Cases ---

TEST_F(FFT_R2C_Forward_Test, KnownTransform_RealDelta_Double) {
  using Real = double;
  using Complex = std::complex<Real>;
  std::vector<Real> input(N, 0.0);
  input[0] = 1.0;  // Real delta function at index 0
  std::vector<Complex> spectrum;
  // Expected for BACKWARD/ORTHO/FORWARD norm (only magnitude changes)
  // RFFT of delta is constant real value (scaled appropriately)
  std::vector<Complex> expected_unscaled(Nc, {1.0, 0.0});

  try {
    // Test convenience function (uses BACKWARD norm -> scale=1)
    OmniDSP::rfft(input, spectrum);
    ExpectComplexVectorNear(expected_unscaled, spectrum, double_tolerance,
                            "rfft(delta) BACKWARD");

    // Test ORTHO norm
    OmniDSP::FFTPlan<Real> plan_ortho(
        N, OmniDSP::Precision::DOUBLE, OmniDSP::Direction::FORWARD,
        OmniDSP::Domain::REAL, OmniDSP::NormMode::ORTHO);
    ASSERT_EQ(plan_ortho.getComplexLength(), Nc);
    spectrum.assign(Nc, {0.0, 0.0});  // Clear spectrum
    plan_ortho.execute_rfft(input.data(), spectrum.data());
    Real ortho_scale = 1.0 / std::sqrt(static_cast<Real>(N));
    std::vector<Complex> expected_ortho(Nc, {ortho_scale, 0.0});
    ExpectComplexVectorNear(expected_ortho, spectrum, double_tolerance,
                            "rfft(delta) ORTHO");

    // Test FORWARD norm
    OmniDSP::FFTPlan<Real> plan_fwd(
        N, OmniDSP::Precision::DOUBLE, OmniDSP::Direction::FORWARD,
        OmniDSP::Domain::REAL, OmniDSP::NormMode::FORWARD);
    ASSERT_EQ(plan_fwd.getComplexLength(), Nc);
    spectrum.assign(Nc, {0.0, 0.0});  // Clear spectrum
    plan_fwd.execute_rfft(input.data(), spectrum.data());
    Real fwd_scale = 1.0 / static_cast<Real>(N);
    std::vector<Complex> expected_fwd(Nc, {fwd_scale, 0.0});
    ExpectComplexVectorNear(expected_fwd, spectrum, double_tolerance,
                            "rfft(delta) FORWARD");

  } catch (const std::exception& e) {
    FAIL() << "RFFT operation threw exception: " << e.what();
  }
}

TEST_F(FFT_R2C_Forward_Test, KnownTransform_RealDC_Float) {
  using Real = float;
  using Complex = std::complex<Real>;
  std::vector<Real> input(N, 1.0f);  // Real DC signal
  std::vector<Complex> spectrum;
  // Expected for BACKWARD norm (unscaled forward)
  std::vector<Complex> expected(Nc, {0.0f, 0.0f});
  expected[0] = {static_cast<Real>(N), 0.0f};  // RFFT of DC=1 is N at freq 0

  try {
    OmniDSP::rfft(input, spectrum);
    ASSERT_EQ(spectrum.size(), Nc);
    ExpectComplexVectorNear(expected, spectrum, float_tolerance, "rfft(DC)");
  } catch (const std::exception& e) {
    FAIL() << "RFFT operation threw exception: " << e.what();
  }
}

TEST_F(FFT_R2C_Forward_Test, KnownTransform_RealCosine_Double) {
  using Real = double;
  using Complex = std::complex<Real>;
  std::vector<Real> input(N);
  const double freq = 2.0;  // Example frequency (2 cycles in N samples)
  for (size_t i = 0; i < N; ++i) input[i] = std::cos(2.0 * M_PI * freq * i / N);

  std::vector<Complex> spectrum;
  // Expected for BACKWARD norm (unscaled forward)
  // Cosine -> two peaks at +/- freq index
  std::vector<Complex> expected(Nc, {0.0, 0.0});
  // For cos(2*pi*k*t/N), expect peaks at index k and N-k.
  // Since output is size N/2+1, we only see the peak at index k.
  // Amplitude should be N/2 for cosine.
  expected[static_cast<size_t>(freq)] = {static_cast<Real>(N) / 2.0, 0.0};

  try {
    OmniDSP::rfft(input, spectrum);
    ASSERT_EQ(spectrum.size(), Nc);
    ExpectComplexVectorNear(expected, spectrum, double_tolerance,
                            "rfft(Cosine)");
  } catch (const std::exception& e) {
    FAIL() << "RFFT operation threw exception: " << e.what();
  }
}

// TODO: Add more tests:
// - Check Hermitian symmetry of output explicitly
// - Different power-of-2 lengths
// - Edge case N=1
// - Error condition tests (non-power-of-2 for Accelerate, wrong
// domain/direction plan)

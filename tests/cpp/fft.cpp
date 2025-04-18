// tests/fft.cpp
// Tests for FFT functions (fft, ifft, rfft, irfft) using reference data.
// Covers convenience functions and FFTPlan with different norms.

#include <OmniDSP/omnidsp.h>  // For FFT functions and FFTPlan
#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <limits>     // For numeric_limits
#include <stdexcept>  // For exception handling
#include <vector>

#include "test_data_utils.h"  // Use the data loading utility

// NOTE: Relying on WIN32_LEAN_AND_MEAN and NOMINMAX defined in CMakeLists.txt
//       to prevent FORWARD/BACKWARD/FLOAT/min/max macro collisions on Windows.

// --- Test Fixture ---
class FFT_Test : public ::testing::Test {
 protected:
  // FFT length matches the one used in generate_references.py
  const size_t N = 16;
  // Tolerances might need adjustment based on backend precision
  const double double_tolerance =
      std::numeric_limits<double>::epsilon() * N * 100;
  const float float_tolerance = std::numeric_limits<float>::epsilon() * N * 100;

  // Helper to compare complex vectors with tolerance
  template <typename T>
  void ExpectComplexVectorNear(const std::vector<std::complex<T>> &expected,
                               const std::vector<std::complex<T>> &actual,
                               T tolerance, const std::string &msg = "") {
    ASSERT_EQ(expected.size(), actual.size()) << msg << " - Size mismatch";
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(expected[i].real(), actual[i].real(), tolerance)
          << msg << " - Mismatch at index " << i << " (real)";
      EXPECT_NEAR(expected[i].imag(), actual[i].imag(), tolerance)
          << msg << " - Mismatch at index " << i << " (imag)";
    }
  }

  // Helper to compare real vectors with tolerance
  template <typename T>
  void ExpectRealVectorNear(const std::vector<T> &expected,
                            const std::vector<T> &actual, T tolerance,
                            const std::string &msg = "") {
    ASSERT_EQ(expected.size(), actual.size()) << msg << " - Size mismatch";
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(expected[i], actual[i], tolerance)
          << msg << " - Mismatch at index " << i;
    }
  }

  // Helper function to convert vector<complex<P>> to vector<complex<T>>
  // Needed because reference data might be stored as double complex even for
  // float tests. Assumes P is double, T is float or double.
  template <typename T, typename P = double>
  std::vector<std::complex<T>> convertComplexVector(
      const std::vector<std::complex<P>> &source) {
    std::vector<std::complex<T>> dest(source.size());
    for (size_t i = 0; i < source.size(); ++i) {
      dest[i] = std::complex<T>(static_cast<T>(source[i].real()),
                                static_cast<T>(source[i].imag()));
    }
    return dest;
  }
};

// --- FFT (Complex Forward) Tests ---

TEST_F(FFT_Test, Convenience_FFT_Double) {
  using Complex = std::complex<double>;
  const auto &input =
      TestDataUtils::getExpectedComplexDoubleVec("FFT_INPUT_COMPLEX_D");
  const auto &expected =
      TestDataUtils::getExpectedComplexDoubleVec("FFT_EXPECTED_BACKWARD_D");
  std::vector<Complex> spectrum;

  ASSERT_NO_THROW(OmniDSP::fft(input, spectrum));
  ExpectComplexVectorNear(expected, spectrum, double_tolerance, "fft(D)");
}

TEST_F(FFT_Test, Convenience_FFT_Float) {
  using Complex = std::complex<float>;
  const auto &input =
      TestDataUtils::getExpectedComplexFloatVec("FFT_INPUT_COMPLEX_F");
  const auto &expected =
      TestDataUtils::getExpectedComplexFloatVec("FFT_EXPECTED_BACKWARD_F");
  std::vector<Complex> spectrum;

  ASSERT_NO_THROW(OmniDSP::fft(input, spectrum));
  ExpectComplexVectorNear(expected, spectrum, float_tolerance, "fft(F)");
}

TEST_F(FFT_Test, Plan_FFT_Backward_Double) {
  using Complex = std::complex<double>;
  const auto &input =
      TestDataUtils::getExpectedComplexDoubleVec("FFT_INPUT_COMPLEX_D");
  const auto &expected =
      TestDataUtils::getExpectedComplexDoubleVec("FFT_EXPECTED_BACKWARD_D");
  std::vector<Complex> spectrum(N);

  OmniDSP::FFTPlan<double> plan(
      N, OmniDSP::Precision::DOUBLE, OmniDSP::Direction::FORWARD,
      OmniDSP::Domain::COMPLEX, OmniDSP::NormMode::BACKWARD);
  ASSERT_NO_THROW(plan.execute(input.data(), spectrum.data()));
  ExpectComplexVectorNear(expected, spectrum, double_tolerance,
                          "FFTPlan BACKWARD(D)");
}

TEST_F(FFT_Test, Plan_FFT_Backward_Float) {
  using Complex = std::complex<float>;
  const auto &input =
      TestDataUtils::getExpectedComplexFloatVec("FFT_INPUT_COMPLEX_F");
  const auto &expected =
      TestDataUtils::getExpectedComplexFloatVec("FFT_EXPECTED_BACKWARD_F");
  std::vector<Complex> spectrum(N);

  OmniDSP::FFTPlan<float> plan(
      N, OmniDSP::Precision::SINGLE, OmniDSP::Direction::FORWARD,
      OmniDSP::Domain::COMPLEX, OmniDSP::NormMode::BACKWARD);
  ASSERT_NO_THROW(plan.execute(input.data(), spectrum.data()));
  ExpectComplexVectorNear(expected, spectrum, float_tolerance,
                          "FFTPlan BACKWARD(F)");
}

TEST_F(FFT_Test, Plan_FFT_Ortho_Double) {
  using Complex = std::complex<double>;
  const auto &input =
      TestDataUtils::getExpectedComplexDoubleVec("FFT_INPUT_COMPLEX_D");
  const auto &expected =
      TestDataUtils::getExpectedComplexDoubleVec("FFT_EXPECTED_ORTHO_D");
  std::vector<Complex> spectrum(N);

  OmniDSP::FFTPlan<double> plan(
      N, OmniDSP::Precision::DOUBLE, OmniDSP::Direction::FORWARD,
      OmniDSP::Domain::COMPLEX, OmniDSP::NormMode::ORTHO);
  ASSERT_NO_THROW(plan.execute(input.data(), spectrum.data()));
  ExpectComplexVectorNear(expected, spectrum, double_tolerance,
                          "FFTPlan ORTHO(D)");
}

TEST_F(FFT_Test, Plan_FFT_Ortho_Float) {
  using Complex = std::complex<float>;
  const auto &input =
      TestDataUtils::getExpectedComplexFloatVec("FFT_INPUT_COMPLEX_F");
  const auto &expected =
      TestDataUtils::getExpectedComplexFloatVec("FFT_EXPECTED_ORTHO_F");
  std::vector<Complex> spectrum(N);

  OmniDSP::FFTPlan<float> plan(
      N, OmniDSP::Precision::SINGLE, OmniDSP::Direction::FORWARD,
      OmniDSP::Domain::COMPLEX, OmniDSP::NormMode::ORTHO);
  ASSERT_NO_THROW(plan.execute(input.data(), spectrum.data()));
  ExpectComplexVectorNear(expected, spectrum, float_tolerance,
                          "FFTPlan ORTHO(F)");
}

TEST_F(FFT_Test, Plan_FFT_Forward_Double) {
  using Complex = std::complex<double>;
  const auto &input =
      TestDataUtils::getExpectedComplexDoubleVec("FFT_INPUT_COMPLEX_D");
  const auto &expected =
      TestDataUtils::getExpectedComplexDoubleVec("FFT_EXPECTED_FORWARD_D");
  std::vector<Complex> spectrum(N);

  OmniDSP::FFTPlan<double> plan(
      N, OmniDSP::Precision::DOUBLE, OmniDSP::Direction::FORWARD,
      OmniDSP::Domain::COMPLEX, OmniDSP::NormMode::FORWARD);
  ASSERT_NO_THROW(plan.execute(input.data(), spectrum.data()));
  ExpectComplexVectorNear(expected, spectrum, double_tolerance,
                          "FFTPlan FORWARD(D)");
}

TEST_F(FFT_Test, Plan_FFT_Forward_Float) {
  using Complex = std::complex<float>;
  const auto &input =
      TestDataUtils::getExpectedComplexFloatVec("FFT_INPUT_COMPLEX_F");
  const auto &expected =
      TestDataUtils::getExpectedComplexFloatVec("FFT_EXPECTED_FORWARD_F");
  std::vector<Complex> spectrum(N);

  OmniDSP::FFTPlan<float> plan(
      N, OmniDSP::Precision::SINGLE, OmniDSP::Direction::FORWARD,
      OmniDSP::Domain::COMPLEX, OmniDSP::NormMode::FORWARD);
  ASSERT_NO_THROW(plan.execute(input.data(), spectrum.data()));
  ExpectComplexVectorNear(expected, spectrum, float_tolerance,
                          "FFTPlan FORWARD(F)");
}

// --- IFFT (Complex Inverse) Tests ---

TEST_F(FFT_Test, Convenience_IFFT_Double) {
  using Complex = std::complex<double>;
  const auto &input_spectrum =
      TestDataUtils::getExpectedComplexDoubleVec("FFT_EXPECTED_BACKWARD_D");
  const auto &expected_signal =
      TestDataUtils::getExpectedComplexDoubleVec("IFFT_EXPECTED_BACKWARD_D");
  std::vector<Complex> time_signal;

  ASSERT_NO_THROW(OmniDSP::ifft(input_spectrum, time_signal));
  ExpectComplexVectorNear(expected_signal, time_signal, double_tolerance,
                          "ifft(D)");
}

TEST_F(FFT_Test, Convenience_IFFT_Float) {
  using Complex = std::complex<float>;
  const auto &input_spectrum =
      TestDataUtils::getExpectedComplexFloatVec("FFT_EXPECTED_BACKWARD_F");
  const auto &expected_signal =
      TestDataUtils::getExpectedComplexFloatVec("IFFT_EXPECTED_BACKWARD_F");
  std::vector<Complex> time_signal;

  ASSERT_NO_THROW(OmniDSP::ifft(input_spectrum, time_signal));
  ExpectComplexVectorNear(expected_signal, time_signal, float_tolerance,
                          "ifft(F)");
}

TEST_F(FFT_Test, Plan_IFFT_Backward_Double) {
  using Complex = std::complex<double>;
  const auto &input_spectrum =
      TestDataUtils::getExpectedComplexDoubleVec("FFT_EXPECTED_BACKWARD_D");
  const auto &expected_signal =
      TestDataUtils::getExpectedComplexDoubleVec("IFFT_EXPECTED_BACKWARD_D");
  std::vector<Complex> time_signal(N);

  OmniDSP::FFTPlan<double> plan(
      N, OmniDSP::Precision::DOUBLE, OmniDSP::Direction::INVERSE,
      OmniDSP::Domain::COMPLEX, OmniDSP::NormMode::BACKWARD);
  ASSERT_NO_THROW(plan.execute(input_spectrum.data(), time_signal.data()));
  ExpectComplexVectorNear(expected_signal, time_signal, double_tolerance,
                          "IFFTPlan BACKWARD(D)");
}

TEST_F(FFT_Test, Plan_IFFT_Backward_Float) {
  using Complex = std::complex<float>;
  const auto &input_spectrum =
      TestDataUtils::getExpectedComplexFloatVec("FFT_EXPECTED_BACKWARD_F");
  const auto &expected_signal =
      TestDataUtils::getExpectedComplexFloatVec("IFFT_EXPECTED_BACKWARD_F");
  std::vector<Complex> time_signal(N);

  OmniDSP::FFTPlan<float> plan(
      N, OmniDSP::Precision::SINGLE, OmniDSP::Direction::INVERSE,
      OmniDSP::Domain::COMPLEX, OmniDSP::NormMode::BACKWARD);
  ASSERT_NO_THROW(plan.execute(input_spectrum.data(), time_signal.data()));
  ExpectComplexVectorNear(expected_signal, time_signal, float_tolerance,
                          "IFFTPlan BACKWARD(F)");
}

TEST_F(FFT_Test, Plan_IFFT_Ortho_Double) {
  using Complex = std::complex<double>;
  const auto &input_spectrum =
      TestDataUtils::getExpectedComplexDoubleVec("FFT_EXPECTED_ORTHO_D");
  const auto &expected_signal =
      TestDataUtils::getExpectedComplexDoubleVec("IFFT_EXPECTED_ORTHO_D");
  std::vector<Complex> time_signal(N);

  OmniDSP::FFTPlan<double> plan(
      N, OmniDSP::Precision::DOUBLE, OmniDSP::Direction::INVERSE,
      OmniDSP::Domain::COMPLEX, OmniDSP::NormMode::ORTHO);
  ASSERT_NO_THROW(plan.execute(input_spectrum.data(), time_signal.data()));
  ExpectComplexVectorNear(expected_signal, time_signal, double_tolerance,
                          "IFFTPlan ORTHO(D)");
}

TEST_F(FFT_Test, Plan_IFFT_Ortho_Float) {
  using Complex = std::complex<float>;
  const auto &input_spectrum =
      TestDataUtils::getExpectedComplexFloatVec("FFT_EXPECTED_ORTHO_F");
  const auto &expected_signal =
      TestDataUtils::getExpectedComplexFloatVec("IFFT_EXPECTED_ORTHO_F");
  std::vector<Complex> time_signal(N);

  OmniDSP::FFTPlan<float> plan(
      N, OmniDSP::Precision::SINGLE, OmniDSP::Direction::INVERSE,
      OmniDSP::Domain::COMPLEX, OmniDSP::NormMode::ORTHO);
  ASSERT_NO_THROW(plan.execute(input_spectrum.data(), time_signal.data()));
  ExpectComplexVectorNear(expected_signal, time_signal, float_tolerance,
                          "IFFTPlan ORTHO(F)");
}

TEST_F(FFT_Test, Plan_IFFT_Forward_Double) {
  using Complex = std::complex<double>;
  const auto &input_spectrum =
      TestDataUtils::getExpectedComplexDoubleVec("FFT_EXPECTED_FORWARD_D");
  const auto &expected_signal =
      TestDataUtils::getExpectedComplexDoubleVec("IFFT_EXPECTED_FORWARD_D");
  std::vector<Complex> time_signal(N);

  OmniDSP::FFTPlan<double> plan(
      N, OmniDSP::Precision::DOUBLE, OmniDSP::Direction::INVERSE,
      OmniDSP::Domain::COMPLEX, OmniDSP::NormMode::FORWARD);
  ASSERT_NO_THROW(plan.execute(input_spectrum.data(), time_signal.data()));
  ExpectComplexVectorNear(expected_signal, time_signal, double_tolerance,
                          "IFFTPlan FORWARD(D)");
}

TEST_F(FFT_Test, Plan_IFFT_Forward_Float) {
  using Complex = std::complex<float>;
  const auto &input_spectrum =
      TestDataUtils::getExpectedComplexFloatVec("FFT_EXPECTED_FORWARD_F");
  const auto &expected_signal =
      TestDataUtils::getExpectedComplexFloatVec("IFFT_EXPECTED_FORWARD_F");
  std::vector<Complex> time_signal(N);

  OmniDSP::FFTPlan<float> plan(
      N, OmniDSP::Precision::SINGLE, OmniDSP::Direction::INVERSE,
      OmniDSP::Domain::COMPLEX, OmniDSP::NormMode::FORWARD);
  ASSERT_NO_THROW(plan.execute(input_spectrum.data(), time_signal.data()));
  ExpectComplexVectorNear(expected_signal, time_signal, float_tolerance,
                          "IFFTPlan FORWARD(F)");
}

// --- RFFT (Real Forward) Tests ---

TEST_F(FFT_Test, Convenience_RFFT_Double) {
  using Real = double;
  using Complex = std::complex<Real>;
  const auto &input = TestDataUtils::getExpectedDoubleVec("FFT_INPUT_REAL_D");
  const auto &expected =
      TestDataUtils::getExpectedComplexDoubleVec("RFFT_EXPECTED_BACKWARD_D");
  std::vector<Complex> spectrum;

  ASSERT_NO_THROW(OmniDSP::rfft(input, spectrum));
  ASSERT_EQ(spectrum.size(), N / 2 + 1);
  ExpectComplexVectorNear(expected, spectrum, double_tolerance, "rfft(D)");
}

TEST_F(FFT_Test, Convenience_RFFT_Float) {
  using Real = float;
  using Complex = std::complex<Real>;
  const auto &input = TestDataUtils::getExpectedFloatVec("FFT_INPUT_REAL_F");
  const auto &expected =
      TestDataUtils::getExpectedComplexFloatVec("RFFT_EXPECTED_BACKWARD_F");
  std::vector<Complex> spectrum;

  ASSERT_NO_THROW(OmniDSP::rfft(input, spectrum));
  ASSERT_EQ(spectrum.size(), N / 2 + 1);
  ExpectComplexVectorNear(expected, spectrum, float_tolerance, "rfft(F)");
}

TEST_F(FFT_Test, Plan_RFFT_Backward_Double) {
  using Real = double;
  using Complex = std::complex<Real>;
  const auto &input = TestDataUtils::getExpectedDoubleVec("FFT_INPUT_REAL_D");
  const auto &expected =
      TestDataUtils::getExpectedComplexDoubleVec("RFFT_EXPECTED_BACKWARD_D");
  std::vector<Complex> spectrum(N / 2 + 1);

  OmniDSP::FFTPlan<Real> plan(
      N, OmniDSP::Precision::DOUBLE, OmniDSP::Direction::FORWARD,
      OmniDSP::Domain::REAL, OmniDSP::NormMode::BACKWARD);
  ASSERT_NO_THROW(plan.execute_rfft(input.data(), spectrum.data()));
  ExpectComplexVectorNear(expected, spectrum, double_tolerance,
                          "RFFTPlan BACKWARD(D)");
}

TEST_F(FFT_Test, Plan_RFFT_Backward_Float) {
  using Real = float;
  using Complex = std::complex<Real>;
  const auto &input = TestDataUtils::getExpectedFloatVec("FFT_INPUT_REAL_F");
  const auto &expected =
      TestDataUtils::getExpectedComplexFloatVec("RFFT_EXPECTED_BACKWARD_F");
  std::vector<Complex> spectrum(N / 2 + 1);

  OmniDSP::FFTPlan<Real> plan(
      N, OmniDSP::Precision::SINGLE, OmniDSP::Direction::FORWARD,
      OmniDSP::Domain::REAL, OmniDSP::NormMode::BACKWARD);
  ASSERT_NO_THROW(plan.execute_rfft(input.data(), spectrum.data()));
  ExpectComplexVectorNear(expected, spectrum, float_tolerance,
                          "RFFTPlan BACKWARD(F)");
}

TEST_F(FFT_Test, Plan_RFFT_Ortho_Double) {
  using Real = double;
  using Complex = std::complex<Real>;
  const auto &input = TestDataUtils::getExpectedDoubleVec("FFT_INPUT_REAL_D");
  const auto &expected =
      TestDataUtils::getExpectedComplexDoubleVec("RFFT_EXPECTED_ORTHO_D");
  std::vector<Complex> spectrum(N / 2 + 1);

  OmniDSP::FFTPlan<Real> plan(N, OmniDSP::Precision::DOUBLE,
                              OmniDSP::Direction::FORWARD,
                              OmniDSP::Domain::REAL, OmniDSP::NormMode::ORTHO);
  ASSERT_NO_THROW(plan.execute_rfft(input.data(), spectrum.data()));
  ExpectComplexVectorNear(expected, spectrum, double_tolerance,
                          "RFFTPlan ORTHO(D)");
}

TEST_F(FFT_Test, Plan_RFFT_Ortho_Float) {
  using Real = float;
  using Complex = std::complex<Real>;
  const auto &input = TestDataUtils::getExpectedFloatVec("FFT_INPUT_REAL_F");
  const auto &expected =
      TestDataUtils::getExpectedComplexFloatVec("RFFT_EXPECTED_ORTHO_F");
  std::vector<Complex> spectrum(N / 2 + 1);

  OmniDSP::FFTPlan<Real> plan(N, OmniDSP::Precision::SINGLE,
                              OmniDSP::Direction::FORWARD,
                              OmniDSP::Domain::REAL, OmniDSP::NormMode::ORTHO);
  ASSERT_NO_THROW(plan.execute_rfft(input.data(), spectrum.data()));
  ExpectComplexVectorNear(expected, spectrum, float_tolerance,
                          "RFFTPlan ORTHO(F)");
}

TEST_F(FFT_Test, Plan_RFFT_Forward_Double) {
  using Real = double;
  using Complex = std::complex<Real>;
  const auto &input = TestDataUtils::getExpectedDoubleVec("FFT_INPUT_REAL_D");
  const auto &expected =
      TestDataUtils::getExpectedComplexDoubleVec("RFFT_EXPECTED_FORWARD_D");
  std::vector<Complex> spectrum(N / 2 + 1);

  OmniDSP::FFTPlan<Real> plan(
      N, OmniDSP::Precision::DOUBLE, OmniDSP::Direction::FORWARD,
      OmniDSP::Domain::REAL, OmniDSP::NormMode::FORWARD);
  ASSERT_NO_THROW(plan.execute_rfft(input.data(), spectrum.data()));
  ExpectComplexVectorNear(expected, spectrum, double_tolerance,
                          "RFFTPlan FORWARD(D)");
}

TEST_F(FFT_Test, Plan_RFFT_Forward_Float) {
  using Real = float;
  using Complex = std::complex<Real>;
  const auto &input = TestDataUtils::getExpectedFloatVec("FFT_INPUT_REAL_F");
  const auto &expected =
      TestDataUtils::getExpectedComplexFloatVec("RFFT_EXPECTED_FORWARD_F");
  std::vector<Complex> spectrum(N / 2 + 1);

  OmniDSP::FFTPlan<Real> plan(
      N, OmniDSP::Precision::SINGLE, OmniDSP::Direction::FORWARD,
      OmniDSP::Domain::REAL, OmniDSP::NormMode::FORWARD);
  ASSERT_NO_THROW(plan.execute_rfft(input.data(), spectrum.data()));
  ExpectComplexVectorNear(expected, spectrum, float_tolerance,
                          "RFFTPlan FORWARD(F)");
}

// --- IRFFT (Real Inverse) Tests ---

TEST_F(FFT_Test, Convenience_IRFFT_Double) {
  using Real = double;
  using Complex = std::complex<Real>;
  const auto &input_spectrum =
      TestDataUtils::getExpectedComplexDoubleVec("RFFT_EXPECTED_BACKWARD_D");
  const auto &expected_signal =
      TestDataUtils::getExpectedDoubleVec("IRFFT_EXPECTED_BACKWARD_D");
  std::vector<Real> time_signal;

  // Corrected: Remove 3rd argument 'N'
  ASSERT_NO_THROW(OmniDSP::irfft(input_spectrum,
                                 time_signal));  // Error was here (Line 388)
  ExpectRealVectorNear(expected_signal, time_signal, double_tolerance,
                       "irfft(D)");
}

TEST_F(FFT_Test, Convenience_IRFFT_Float) {
  using Real = float;
  using Complex = std::complex<Real>;
  const auto &input_spectrum =
      TestDataUtils::getExpectedComplexFloatVec("RFFT_EXPECTED_BACKWARD_F");
  const auto &expected_signal =
      TestDataUtils::getExpectedFloatVec("IRFFT_EXPECTED_BACKWARD_F");
  std::vector<Real> time_signal;

  // Corrected: Remove 3rd argument 'N'
  ASSERT_NO_THROW(OmniDSP::irfft(input_spectrum,
                                 time_signal));  // Error was here (Line 399)
  ExpectRealVectorNear(expected_signal, time_signal, float_tolerance,
                       "irfft(F)");
}

TEST_F(FFT_Test, Plan_IRFFT_Backward_Double) {
  using Real = double;
  using Complex = std::complex<Real>;
  const auto &input_spectrum =
      TestDataUtils::getExpectedComplexDoubleVec("RFFT_EXPECTED_BACKWARD_D");
  const auto &expected_signal =
      TestDataUtils::getExpectedDoubleVec("IRFFT_EXPECTED_BACKWARD_D");
  std::vector<Real> time_signal(N);

  OmniDSP::FFTPlan<Real> plan(
      N, OmniDSP::Precision::DOUBLE, OmniDSP::Direction::INVERSE,
      OmniDSP::Domain::REAL, OmniDSP::NormMode::BACKWARD);
  ASSERT_NO_THROW(
      plan.execute_irfft(input_spectrum.data(), time_signal.data()));
  ExpectRealVectorNear(expected_signal, time_signal, double_tolerance,
                       "IRFFTPlan BACKWARD(D)");
}

TEST_F(FFT_Test, Plan_IRFFT_Backward_Float) {
  using Real = float;
  using Complex = std::complex<Real>;
  const auto &input_spectrum =
      TestDataUtils::getExpectedComplexFloatVec("RFFT_EXPECTED_BACKWARD_F");
  const auto &expected_signal =
      TestDataUtils::getExpectedFloatVec("IRFFT_EXPECTED_BACKWARD_F");
  std::vector<Real> time_signal(N);

  OmniDSP::FFTPlan<Real> plan(
      N, OmniDSP::Precision::SINGLE, OmniDSP::Direction::INVERSE,
      OmniDSP::Domain::REAL, OmniDSP::NormMode::BACKWARD);
  ASSERT_NO_THROW(
      plan.execute_irfft(input_spectrum.data(), time_signal.data()));
  ExpectRealVectorNear(expected_signal, time_signal, float_tolerance,
                       "IRFFTPlan BACKWARD(F)");
}

TEST_F(FFT_Test, Plan_IRFFT_Ortho_Double) {
  using Real = double;
  using Complex = std::complex<Real>;
  const auto &input_spectrum =
      TestDataUtils::getExpectedComplexDoubleVec("RFFT_EXPECTED_ORTHO_D");
  const auto &expected_signal =
      TestDataUtils::getExpectedDoubleVec("IRFFT_EXPECTED_ORTHO_D");
  std::vector<Real> time_signal(N);

  OmniDSP::FFTPlan<Real> plan(N, OmniDSP::Precision::DOUBLE,
                              OmniDSP::Direction::INVERSE,
                              OmniDSP::Domain::REAL, OmniDSP::NormMode::ORTHO);
  ASSERT_NO_THROW(
      plan.execute_irfft(input_spectrum.data(), time_signal.data()));
  ExpectRealVectorNear(expected_signal, time_signal, double_tolerance,
                       "IRFFTPlan ORTHO(D)");
}

TEST_F(FFT_Test, Plan_IRFFT_Ortho_Float) {
  using Real = float;
  using Complex = std::complex<Real>;
  const auto &input_spectrum =
      TestDataUtils::getExpectedComplexFloatVec("RFFT_EXPECTED_ORTHO_F");
  const auto &expected_signal =
      TestDataUtils::getExpectedFloatVec("IRFFT_EXPECTED_ORTHO_F");
  std::vector<Real> time_signal(N);

  OmniDSP::FFTPlan<Real> plan(N, OmniDSP::Precision::SINGLE,
                              OmniDSP::Direction::INVERSE,
                              OmniDSP::Domain::REAL, OmniDSP::NormMode::ORTHO);
  ASSERT_NO_THROW(
      plan.execute_irfft(input_spectrum.data(), time_signal.data()));
  ExpectRealVectorNear(expected_signal, time_signal, float_tolerance,
                       "IRFFTPlan ORTHO(F)");
}

TEST_F(FFT_Test, Plan_IRFFT_Forward_Double) {
  using Real = double;
  using Complex = std::complex<Real>;
  const auto &input_spectrum =
      TestDataUtils::getExpectedComplexDoubleVec("RFFT_EXPECTED_FORWARD_D");
  const auto &expected_signal =
      TestDataUtils::getExpectedDoubleVec("IRFFT_EXPECTED_FORWARD_D");
  std::vector<Real> time_signal(N);

  OmniDSP::FFTPlan<Real> plan(
      N, OmniDSP::Precision::DOUBLE, OmniDSP::Direction::INVERSE,
      OmniDSP::Domain::REAL, OmniDSP::NormMode::FORWARD);
  ASSERT_NO_THROW(
      plan.execute_irfft(input_spectrum.data(), time_signal.data()));
  ExpectRealVectorNear(expected_signal, time_signal, double_tolerance,
                       "IRFFTPlan FORWARD(D)");
}

TEST_F(FFT_Test, Plan_IRFFT_Forward_Float) {
  using Real = float;
  using Complex = std::complex<Real>;
  const auto &input_spectrum =
      TestDataUtils::getExpectedComplexFloatVec("RFFT_EXPECTED_FORWARD_F");
  const auto &expected_signal =
      TestDataUtils::getExpectedFloatVec("IRFFT_EXPECTED_FORWARD_F");
  std::vector<Real> time_signal(N);

  OmniDSP::FFTPlan<Real> plan(
      N, OmniDSP::Precision::SINGLE, OmniDSP::Direction::INVERSE,
      OmniDSP::Domain::REAL, OmniDSP::NormMode::FORWARD);
  ASSERT_NO_THROW(
      plan.execute_irfft(input_spectrum.data(), time_signal.data()));
  ExpectRealVectorNear(expected_signal, time_signal, float_tolerance,
                       "IRFFTPlan FORWARD(F)");
}

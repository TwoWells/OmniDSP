// tests/ifft.cpp
// Tests for inverse Complex-to-Complex FFT functions (ifft, ifft_inplace, FFTPlan::execute)

#include <gtest/gtest.h>
#include <OmniFFT/omnifft.h> // Use correct include path
#include <vector>
#include <complex>
#include <limits>
#include <cmath>

// --- Test Fixture ---
class FFT_C2C_Inverse_Test : public ::testing::Test {
protected:
    const size_t N = 16;
    const double double_tolerance = std::numeric_limits<double>::epsilon() * N * 10;
    const float float_tolerance = std::numeric_limits<float>::epsilon() * N * 10;

    // Helper to compare complex vectors (copied from fft.cpp - consider common header)
    template<typename T>
    void ExpectComplexVectorNear(const std::vector<std::complex<T>>& expected,
                                 const std::vector<std::complex<T>>& actual,
                                 T tolerance, const std::string& msg = "") {
        ASSERT_EQ(expected.size(), actual.size()) << msg << " - Size mismatch";
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(expected[i].real(), actual[i].real(), tolerance) << msg << " - Mismatch at index " << i << " (real)";
            EXPECT_NEAR(expected[i].imag(), actual[i].imag(), tolerance) << msg << " - Mismatch at index " << i << " (imag)";
        }
    }

    // Helper to generate a test signal
    template<typename T>
    std::vector<std::complex<T>> generate_signal(size_t len) {
         std::vector<std::complex<T>> signal(len);
         for (size_t i = 0; i < len; ++i) {
            signal[i] = std::complex<T>(std::sin(2.0 * M_PI * i / len),
                                        std::cos(4.0 * M_PI * i / len));
        }
        return signal;
    }
};

// --- Test Cases ---

TEST_F(FFT_C2C_Inverse_Test, Identity_BackwardNorm_Double) {
    using Complex = std::complex<double>;
    std::vector<Complex> original = generate_signal<double>(N);
    std::vector<Complex> spectrum(N);
    std::vector<Complex> reconstructed(N);

    try {
        // Use BACKWARD norm for both forward and inverse (default for convenience funcs)
        OmniFFT::fft(original, spectrum);
        OmniFFT::ifft(spectrum, reconstructed);
        // With BACKWARD norm, IFFT(FFT(x)) should equal x directly
        ExpectComplexVectorNear(original, reconstructed, double_tolerance, "fft/ifft BACKWARD");

        // Test Plan version
        OmniFFT::FFTPlan<double> fwd_plan(N, OmniFFT::Precision::DOUBLE, OmniFFT::Direction::FORWARD, OmniFFT::Domain::COMPLEX, OmniFFT::NormMode::BACKWARD);
        OmniFFT::FFTPlan<double> inv_plan(N, OmniFFT::Precision::DOUBLE, OmniFFT::Direction::INVERSE, OmniFFT::Domain::COMPLEX, OmniFFT::NormMode::BACKWARD);
        fwd_plan.execute(original.data(), spectrum.data());
        inv_plan.execute(spectrum.data(), reconstructed.data());
        ExpectComplexVectorNear(original, reconstructed, double_tolerance, "FFTPlan BACKWARD");

    } catch (const std::exception& e) {
        FAIL() << "FFT operation threw exception: " << e.what();
    }
}

TEST_F(FFT_C2C_Inverse_Test, Identity_OrthoNorm_Float) {
    using Complex = std::complex<float>;
    std::vector<Complex> original = generate_signal<float>(N);
    std::vector<Complex> spectrum(N);
    std::vector<Complex> reconstructed(N);
    const float tol = float_tolerance * 10; // Ortho might have slightly more error?

    try {
        // Use ORTHO norm for both forward and inverse plans
        OmniFFT::FFTPlan<float> fwd_plan(N, OmniFFT::Precision::SINGLE, OmniFFT::Direction::FORWARD, OmniFFT::Domain::COMPLEX, OmniFFT::NormMode::ORTHO);
        OmniFFT::FFTPlan<float> inv_plan(N, OmniFFT::Precision::SINGLE, OmniFFT::Direction::INVERSE, OmniFFT::Domain::COMPLEX, OmniFFT::NormMode::ORTHO);

        fwd_plan.execute(original.data(), spectrum.data());
        inv_plan.execute(spectrum.data(), reconstructed.data());
        // With ORTHO norm, IFFT(FFT(x)) should also equal x directly
        ExpectComplexVectorNear(original, reconstructed, tol, "FFTPlan ORTHO");

    } catch (const std::exception& e) {
        FAIL() << "FFT operation threw exception: " << e.what();
    }
}

TEST_F(FFT_C2C_Inverse_Test, Identity_ForwardNorm_Double) {
    using Complex = std::complex<double>;
    std::vector<Complex> original = generate_signal<double>(N);
    std::vector<Complex> spectrum(N);
    std::vector<Complex> reconstructed(N);

    try {
        // Use FORWARD norm for both forward and inverse plans
        OmniFFT::FFTPlan<double> fwd_plan(N, OmniFFT::Precision::DOUBLE, OmniFFT::Direction::FORWARD, OmniFFT::Domain::COMPLEX, OmniFFT::NormMode::FORWARD);
        OmniFFT::FFTPlan<double> inv_plan(N, OmniFFT::Precision::DOUBLE, OmniFFT::Direction::INVERSE, OmniFFT::Domain::COMPLEX, OmniFFT::NormMode::FORWARD);

        fwd_plan.execute(original.data(), spectrum.data());
        inv_plan.execute(spectrum.data(), reconstructed.data());
        // With FORWARD norm, IFFT(FFT(x)) should also equal x directly
        ExpectComplexVectorNear(original, reconstructed, double_tolerance, "FFTPlan FORWARD");

    } catch (const std::exception& e) {
        FAIL() << "FFT operation threw exception: " << e.what();
    }
}


TEST_F(FFT_C2C_Inverse_Test, InPlace_Float) {
    using Complex = std::complex<float>;
    std::vector<Complex> original = generate_signal<float>(N);
    std::vector<Complex> data = original; // Copy for in-place testing

    try {
        // Test convenience functions (use BACKWARD norm)
        OmniFFT::fft_inplace(data);
        OmniFFT::ifft_inplace(data);
        ExpectComplexVectorNear(original, data, float_tolerance, "fft_inplace/ifft_inplace");

    } catch (const std::exception& e) {
        FAIL() << "FFT operation threw exception: " << e.what();
    }
}

// TODO: Add more tests:
// - Known spectrum input -> known signal output
// - Different lengths
// - Error condition tests (e.g., wrong domain plan)
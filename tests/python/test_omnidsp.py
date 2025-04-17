# tests/python/test_omnidsp.py
# Updated for overloaded API and factory functions

import unittest
import omnidsp_py  # Import the main package
import numpy as np


# A simple Python window function for CQT tests
def python_hann_window(placeholder_array: np.ndarray) -> np.ndarray:
    # This is a placeholder; CQTPlan determines the actual length needed.
    # For testing the binding, we just need a callable.
    # In a real CQTPlan usage, the C++ side *might* need adaptation
    # if it relies on the input vector size passed to std::function,
    # but pybind11 should handle the basic callable conversion.
    # A more robust C++ CQTPlan might ideally pass the *required* length
    # to the std::function if possible.
    # For now, just return a dummy array of the correct dtype.
    dtype = placeholder_array.dtype
    # Returning a simple Hann window coefficient for size 1 for simplicity
    # The C++ CQTPlan implementation needs to generate the actual window internally.
    return np.array([0.0], dtype=dtype)  # Placeholder for testing callable passing


class TestOmniDSP(unittest.TestCase):
    N = 1024  # Default test size
    N_cqt = 4096  # Longer size often better for CQT

    def test_enums(self):
        """Tests accessing Enum values."""
        self.assertEqual(omnidsp_py.Direction.FORWARD, omnidsp_py.Direction.FORWARD)
        self.assertEqual(omnidsp_py.Precision.SINGLE, omnidsp_py.Precision.SINGLE)
        self.assertEqual(omnidsp_py.Domain.COMPLEX, omnidsp_py.Domain.COMPLEX)
        self.assertEqual(omnidsp_py.NormMode.BACKWARD, omnidsp_py.NormMode.BACKWARD)
        # Add more checks if desired

    def test_fft_plan_factory(self):
        """Tests creating FFT plans using the factory function."""
        # Create Float Plan
        plan_f = omnidsp_py.create_fft_plan(
            length=self.N,
            precision=omnidsp_py.Precision.SINGLE,
            direction=omnidsp_py.Direction.FORWARD,
            domain=omnidsp_py.Domain.COMPLEX,
        )
        self.assertIsInstance(plan_f, omnidsp_py.FFTPlanFloat)
        self.assertEqual(plan_f.getLength(), self.N)
        self.assertEqual(plan_f.getPrecision(), omnidsp_py.Precision.SINGLE)
        self.assertEqual(plan_f.getDirection(), omnidsp_py.Direction.FORWARD)
        self.assertEqual(plan_f.getDomain(), omnidsp_py.Domain.COMPLEX)
        self.assertEqual(
            plan_f.getNormMode(), omnidsp_py.NormMode.BACKWARD
        )  # Default norm

        # Create Double Plan with different options
        plan_d = omnidsp_py.create_fft_plan(
            length=self.N // 2,
            precision=omnidsp_py.Precision.DOUBLE,
            direction=omnidsp_py.Direction.INVERSE,
            domain=omnidsp_py.Domain.REAL,
            norm=omnidsp_py.NormMode.ORTHO,
        )
        self.assertIsInstance(plan_d, omnidsp_py.FFTPlanDouble)
        self.assertEqual(plan_d.getLength(), self.N // 2)
        self.assertEqual(plan_d.getPrecision(), omnidsp_py.Precision.DOUBLE)
        self.assertEqual(plan_d.getDirection(), omnidsp_py.Direction.INVERSE)
        self.assertEqual(plan_d.getDomain(), omnidsp_py.Domain.REAL)
        self.assertEqual(plan_d.getNormMode(), omnidsp_py.NormMode.ORTHO)

    def test_cqt_plan_factory_and_execute(self):
        """Tests creating and executing CQT plans via factory."""
        sr = 22050.0
        fmin = 50.0
        fmax = 5000.0
        bpo = 12

        # Test Double Precision CQT Plan creation
        plan_d = omnidsp_py.create_cqt_plan(
            sample_rate=sr,
            lowest_freq=fmin,
            highest_freq=fmax,
            bins_per_octave=bpo,
            window_function=python_hann_window,  # Pass Python callable
            precision=omnidsp_py.Precision.DOUBLE,
        )
        self.assertIsInstance(plan_d, omnidsp_py.CQTPlanDouble)
        self.assertEqual(plan_d.getSampleRate(), sr)
        self.assertEqual(plan_d.getLowestFrequency(), fmin)
        # Add checks for other properties if needed

        # Test basic execution (Double)
        signal_d = np.sin(2 * np.pi * 440.0 * np.arange(self.N_cqt) / sr).astype(
            np.float64
        )
        cqt_output_d = plan_d.execute(signal_d)
        self.assertEqual(cqt_output_d.ndim, 1)
        self.assertEqual(cqt_output_d.shape[0], plan_d.getNumBins())
        self.assertEqual(cqt_output_d.dtype, np.complex128)

        # Test Float Precision CQT Plan creation
        plan_f = omnidsp_py.create_cqt_plan(
            sample_rate=sr,
            lowest_freq=fmin,
            highest_freq=fmax,
            bins_per_octave=bpo,
            window_function=python_hann_window,  # Pass Python callable
            precision=omnidsp_py.Precision.SINGLE,
        )
        self.assertIsInstance(plan_f, omnidsp_py.CQTPlanFloat)

        # Test basic execution (Float)
        signal_f = signal_d.astype(np.float32)
        cqt_output_f = plan_f.execute(signal_f)
        self.assertEqual(cqt_output_f.ndim, 1)
        self.assertEqual(cqt_output_f.shape[0], plan_f.getNumBins())
        self.assertEqual(cqt_output_f.dtype, np.complex64)

    def test_window_hann(self):
        """Tests overloaded Window.hann static method."""
        # Float
        input_f32 = np.random.rand(self.N).astype(np.float32)
        output_f32 = omnidsp_py.Window.hann(input_f32)
        self.assertEqual(output_f32.shape, (self.N,))
        self.assertEqual(output_f32.dtype, np.float32)
        # Basic check: windowing should reduce magnitude (except potentially ends)
        self.assertTrue(np.all(np.abs(output_f32[1:-1]) <= np.abs(input_f32[1:-1])))

        # Double
        input_f64 = np.random.rand(self.N).astype(np.float64)
        output_f64 = omnidsp_py.Window.hann(input_f64)
        self.assertEqual(output_f64.shape, (self.N,))
        self.assertEqual(output_f64.dtype, np.float64)
        self.assertTrue(np.all(np.abs(output_f64[1:-1]) <= np.abs(input_f64[1:-1])))

    def test_window_hamming(self):
        """Tests overloaded Window.hamming static method."""
        # Float
        input_f32 = np.random.rand(self.N).astype(np.float32)
        output_f32 = omnidsp_py.Window.hamming(input_f32)
        self.assertEqual(output_f32.shape, (self.N,))
        self.assertEqual(output_f32.dtype, np.float32)
        self.assertTrue(np.all(np.abs(output_f32[1:-1]) <= np.abs(input_f32[1:-1])))

        # Double
        input_f64 = np.random.rand(self.N).astype(np.float64)
        output_f64 = omnidsp_py.Window.hamming(input_f64)
        self.assertEqual(output_f64.shape, (self.N,))
        self.assertEqual(output_f64.dtype, np.float64)
        self.assertTrue(np.all(np.abs(output_f64[1:-1]) <= np.abs(input_f64[1:-1])))

    def test_window_kaiser(self):
        """Tests overloaded Window.kaiser static method."""
        beta = 8.6
        # Float
        input_f32 = np.random.rand(self.N).astype(np.float32)
        output_f32 = omnidsp_py.Window.kaiser(input_f32, beta=np.float32(beta))
        self.assertEqual(output_f32.shape, (self.N,))
        self.assertEqual(output_f32.dtype, np.float32)
        self.assertTrue(np.all(np.abs(output_f32[1:-1]) <= np.abs(input_f32[1:-1])))

        # Double
        input_f64 = np.random.rand(self.N).astype(np.float64)
        output_f64 = omnidsp_py.Window.kaiser(input_f64, beta=np.float64(beta))
        self.assertEqual(output_f64.shape, (self.N,))
        self.assertEqual(output_f64.dtype, np.float64)
        self.assertTrue(np.all(np.abs(output_f64[1:-1]) <= np.abs(input_f64[1:-1])))

    def test_window_flattop(self):
        """Tests overloaded Window.flattop static method."""
        # Float
        input_f32 = np.random.rand(self.N).astype(np.float32)
        output_f32 = omnidsp_py.Window.flattop(input_f32)
        self.assertEqual(output_f32.shape, (self.N,))
        self.assertEqual(output_f32.dtype, np.float32)
        # Flattop can have values slightly > 1 near center, just check shape/type

        # Double
        input_f64 = np.random.rand(self.N).astype(np.float64)
        output_f64 = omnidsp_py.Window.flattop(input_f64)
        self.assertEqual(output_f64.shape, (self.N,))
        self.assertEqual(output_f64.dtype, np.float64)

    def test_fft(self):
        """Tests overloaded fft convenience function."""
        # Float
        input_c64 = (np.random.rand(self.N) + 1j * np.random.rand(self.N)).astype(
            np.complex64
        )
        output_c64 = omnidsp_py.fft(input_c64)
        self.assertEqual(output_c64.shape, (self.N,))
        self.assertEqual(output_c64.dtype, np.complex64)

        # Double
        input_c128 = (np.random.rand(self.N) + 1j * np.random.rand(self.N)).astype(
            np.complex128
        )
        output_c128 = omnidsp_py.fft(input_c128)
        self.assertEqual(output_c128.shape, (self.N,))
        self.assertEqual(output_c128.dtype, np.complex128)

        # Check identity (approximately) using default BACKWARD norm
        recon_c128 = omnidsp_py.ifft(output_c128)
        self.assertTrue(np.allclose(input_c128, recon_c128, atol=1e-12))

    def test_ifft(self):
        """Tests overloaded ifft convenience function."""
        # Float
        input_c64 = (np.random.rand(self.N) + 1j * np.random.rand(self.N)).astype(
            np.complex64
        )
        output_c64 = omnidsp_py.ifft(input_c64)
        self.assertEqual(output_c64.shape, (self.N,))
        self.assertEqual(output_c64.dtype, np.complex64)

        # Double
        input_c128 = (np.random.rand(self.N) + 1j * np.random.rand(self.N)).astype(
            np.complex128
        )
        output_c128 = omnidsp_py.ifft(input_c128)
        self.assertEqual(output_c128.shape, (self.N,))
        self.assertEqual(output_c128.dtype, np.complex128)

        # Check identity (approximately) using default BACKWARD norm
        recon_c128 = omnidsp_py.fft(output_c128)
        self.assertTrue(np.allclose(input_c128, recon_c128, atol=1e-12))

    def test_rfft(self):
        """Tests overloaded rfft convenience function."""
        expected_len = self.N // 2 + 1

        # Float
        input_f32 = np.random.rand(self.N).astype(np.float32)
        output_c64 = omnidsp_py.rfft(input_f32)
        self.assertEqual(output_c64.shape, (expected_len,))
        self.assertEqual(output_c64.dtype, np.complex64)

        # Double
        input_f64 = np.random.rand(self.N).astype(np.float64)
        output_c128 = omnidsp_py.rfft(input_f64)
        self.assertEqual(output_c128.shape, (expected_len,))
        self.assertEqual(output_c128.dtype, np.complex128)

        # Check identity (approximately) using default BACKWARD norm
        recon_f64 = omnidsp_py.irfft(output_c128)
        self.assertEqual(recon_f64.shape, (self.N,))  # irfft should restore length
        self.assertTrue(np.allclose(input_f64, recon_f64, atol=1e-12))

    def test_irfft(self):
        """Tests overloaded irfft convenience function."""
        input_len = self.N // 2 + 1
        expected_len = self.N

        # Float
        # Create valid Hermitian input for irfft
        spec_f32 = (np.random.rand(input_len) + 1j * np.random.rand(input_len)).astype(
            np.complex64
        )
        spec_f32[0] = spec_f32[0].real + 0j  # Ensure DC is real
        if self.N % 2 == 0:  # Ensure Nyquist is real if N is even
            spec_f32[-1] = spec_f32[-1].real + 0j
        output_f32 = omnidsp_py.irfft(spec_f32)
        self.assertEqual(output_f32.shape, (expected_len,))
        self.assertEqual(output_f32.dtype, np.float32)

        # Double
        spec_c128 = (np.random.rand(input_len) + 1j * np.random.rand(input_len)).astype(
            np.complex128
        )
        spec_c128[0] = spec_c128[0].real + 0j  # Ensure DC is real
        if self.N % 2 == 0:  # Ensure Nyquist is real if N is even
            spec_c128[-1] = spec_c128[-1].real + 0j
        output_f64 = omnidsp_py.irfft(spec_c128)
        self.assertEqual(output_f64.shape, (expected_len,))
        self.assertEqual(output_f64.dtype, np.float64)

        # Check identity (approximately) using default BACKWARD norm
        # Need to re-create a proper hermitian spectrum for check
        signal_f64 = np.random.rand(self.N).astype(np.float64)
        spectrum_f64 = omnidsp_py.rfft(signal_f64)
        recon_f64 = omnidsp_py.irfft(spectrum_f64)
        self.assertTrue(np.allclose(signal_f64, recon_f64, atol=1e-12))


if __name__ == "__main__":
    unittest.main()

import unittest
import omnidsp
import numpy as np

class TestOmniDSP(unittest.TestCase):

    def test_enums(self):
        self.assertEqual(omnidsp.Direction.FORWARD, omnidsp.Direction.FORWARD)
        self.assertEqual(omnidsp.Direction.INVERSE, omnidsp.Direction.INVERSE)
        self.assertEqual(omnidsp.Precision.SINGLE, omnidsp.Precision.SINGLE)
        self.assertEqual(omnidsp.Precision.DOUBLE, omnidsp.Precision.DOUBLE)
        self.assertEqual(omnidsp.Domain.COMPLEX, omnidsp.Domain.COMPLEX)
        self.assertEqual(omnidsp.Domain.REAL, omnidsp.Domain.REAL)
        self.assertEqual(omnidsp.NormMode.BACKWARD, omnidsp.NormMode.BACKWARD)
        self.assertEqual(omnidsp.NormMode.ORTHO, omnidsp.NormMode.ORTHO)
        self.assertEqual(omnidsp.NormMode.FORWARD, omnidsp.NormMode.FORWARD)

    def test_fft_plan_float(self):
        plan = omnidsp.FFTPlanFloat(length=1024, precision=omnidsp.Precision.SINGLE, direction=omnidsp.Direction.FORWARD, domain=omnidsp.Domain.COMPLEX)
        self.assertEqual(plan.getLength(), 1024)
        self.assertEqual(plan.getPrecision(), omnidsp.Precision.SINGLE)
        self.assertEqual(plan.getDirection(), omnidsp.Direction.FORWARD)
        self.assertEqual(plan.getDomain(), omnidsp.Domain.COMPLEX)
        self.assertEqual(plan.getNormMode(), omnidsp.NormMode.BACKWARD)

    def test_fft_plan_double(self):
        plan = omnidsp.FFTPlanDouble(length=1024, precision=omnidsp.Precision.DOUBLE, direction=omnidsp.Direction.FORWARD, domain=omnidsp.Domain.COMPLEX)
        self.assertEqual(plan.getLength(), 1024)
        self.assertEqual(plan.getPrecision(), omnidsp.Precision.DOUBLE)
        self.assertEqual(plan.getDirection(), omnidsp.Direction.FORWARD)
        self.assertEqual(plan.getDomain(), omnidsp.Domain.COMPLEX)
        self.assertEqual(plan.getNormMode(), omnidsp.NormMode.BACKWARD)

    def test_window_hann_float(self):
        input_array = np.random.rand(1024).astype(np.float32)
        windowed_array = omnidsp.Window.hann_float(input_array)
        self.assertEqual(len(windowed_array), 1024)

    def test_window_hann_double(self):
        input_array = np.random.rand(1024).astype(np.float64)
        windowed_array = omnidsp.Window.hann_double(input_array)
        self.assertEqual(len(windowed_array), 1024)

    def test_fft_float(self):
        input_array = np.random.rand(1024) + 1j * np.random.rand(1024)
        input_array = input_array.astype(np.complex64)
        output_array = omnidsp.fft_float(input_array)
        self.assertEqual(len(output_array), 1024)

    def test_fft_double(self):
        input_array = np.random.rand(1024) + 1j * np.random.rand(1024)
        input_array = input_array.astype(np.complex128)
        output_array = omnidsp.fft_double(input_array)
        self.assertEqual(len(output_array), 1024)

    def test_ifft_float(self):
        input_array = np.random.rand(1024) + 1j * np.random.rand(1024)
        input_array = input_array.astype(np.complex64)
        output_array = omnidsp.ifft_float(input_array)
        self.assertEqual(len(output_array), 1024)

    def test_ifft_double(self):
        input_array = np.random.rand(1024) + 1j * np.random.rand(1024)
        input_array = input_array.astype(np.complex128)
        output_array = omnidsp.ifft_double(input_array)
        self.assertEqual(len(output_array), 1024)

    def test_rfft_float(self):
        input_array = np.random.rand(1024).astype(np.float32)
        output_array = omnidsp.rfft_float(input_array)
        self.assertEqual(len(output_array), 1024 // 2 + 1)

    def test_rfft_double(self):
        input_array = np.random.rand(1024).astype(np.float64)
        output_array = omnidsp.rfft_double(input_array)
        self.assertEqual(len(output_array), 1024 // 2 + 1)

    def test_irfft_float(self):
        input_array = np.random.rand(1024 // 2 + 1) + 1j * np.random.rand(1024 // 2 + 1)
        input_array = input_array.astype(np.complex64)
        output_array = omnidsp.irfft_float(input_array)
        self.assertEqual(len(output_array), 1024)

    def test_irfft_double(self):
        input_array = np.random.rand(1024 // 2 + 1) + 1j * np.random.rand(1024 // 2 + 1)
        input_array = input_array.astype(np.complex128)
        output_array = omnidsp.irfft_double(input_array)
        self.assertEqual(len(output_array), 1024)


if __name__ == '__main__':
    unittest.main()

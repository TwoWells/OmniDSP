import unittest
import omnidsp_py
import numpy as np

class TestOmniDSP(unittest.TestCase):

    def test_enums(self):
        self.assertEqual(omnidsp_py.Direction.FORWARD, omnidsp_py.Direction.FORWARD)
        self.assertEqual(omnidsp_py.Direction.INVERSE, omnidsp_py.Direction.INVERSE)
        self.assertEqual(omnidsp_py.Precision.SINGLE, omnidsp_py.Precision.SINGLE)
        self.assertEqual(omnidsp_py.Precision.DOUBLE, omnidsp_py.Precision.DOUBLE)
        self.assertEqual(omnidsp_py.Domain.COMPLEX, omnidsp_py.Domain.COMPLEX)
        self.assertEqual(omnidsp_py.Domain.REAL, omnidsp_py.Domain.REAL)
        self.assertEqual(omnidsp_py.NormMode.BACKWARD, omnidsp_py.NormMode.BACKWARD)
        self.assertEqual(omnidsp_py.NormMode.ORTHO, omnidsp_py.NormMode.ORTHO)
        self.assertEqual(omnidsp_py.NormMode.FORWARD, omnidsp_py.NormMode.FORWARD)

    def test_fft_plan_float(self):
        plan = omnidsp_py.FFTPlanFloat(length=1024, precision=omnidsp_py.Precision.SINGLE, direction=omnidsp_py.Direction.FORWARD, domain=omnidsp_py.Domain.COMPLEX)
        self.assertEqual(plan.getLength(), 1024)
        self.assertEqual(plan.getPrecision(), omnidsp_py.Precision.SINGLE)
        self.assertEqual(plan.getDirection(), omnidsp_py.Direction.FORWARD)
        self.assertEqual(plan.getDomain(), omnidsp_py.Domain.COMPLEX)
        self.assertEqual(plan.getNormMode(), omnidsp_py.NormMode.BACKWARD)

    def test_fft_plan_double(self):
        plan = omnidsp_py.FFTPlanDouble(length=1024, precision=omnidsp_py.Precision.DOUBLE, direction=omnidsp_py.Direction.FORWARD, domain=omnidsp_py.Domain.COMPLEX)
        self.assertEqual(plan.getLength(), 1024)
        self.assertEqual(plan.getPrecision(), omnidsp_py.Precision.DOUBLE)
        self.assertEqual(plan.getDirection(), omnidsp_py.Direction.FORWARD)
        self.assertEqual(plan.getDomain(), omnidsp_py.Domain.COMPLEX)
        self.assertEqual(plan.getNormMode(), omnidsp_py.NormMode.BACKWARD)

    def test_window_hann_float(self):
        input_array = np.random.rand(1024).astype(np.float32)
        windowed_array = omnidsp_py.Window.hann_float(input_array)
        self.assertEqual(len(windowed_array), 1024)

    def test_window_hann_double(self):
        input_array = np.random.rand(1024).astype(np.float64)
        windowed_array = omnidsp_py.Window.hann_double(input_array)
        self.assertEqual(len(windowed_array), 1024)

    def test_fft_float(self):
        input_array = np.random.rand(1024) + 1j * np.random.rand(1024)
        input_array = input_array.astype(np.complex64)
        output_array = omnidsp_py.fft_float(input_array)
        self.assertEqual(len(output_array), 1024)

    def test_fft_double(self):
        input_array = np.random.rand(1024) + 1j * np.random.rand(1024)
        input_array = input_array.astype(np.complex128)
        output_array = omnidsp_py.fft_double(input_array)
        self.assertEqual(len(output_array), 1024)

    def test_ifft_float(self):
        input_array = np.random.rand(1024) + 1j * np.random.rand(1024)
        input_array = input_array.astype(np.complex64)
        output_array = omnidsp_py.ifft_float(input_array)
        self.assertEqual(len(output_array), 1024)

    def test_ifft_double(self):
        input_array = np.random.rand(1024) + 1j * np.random.rand(1024)
        input_array = input_array.astype(np.complex128)
        output_array = omnidsp_py.ifft_double(input_array)
        self.assertEqual(len(output_array), 1024)

    def test_rfft_float(self):
        input_array = np.random.rand(1024).astype(np.float32)
        output_array = omnidsp_py.rfft_float(input_array)
        self.assertEqual(len(output_array), 1024 // 2 + 1)

    def test_rfft_double(self):
        input_array = np.random.rand(1024).astype(np.float64)
        output_array = omnidsp_py.rfft_double(input_array)
        self.assertEqual(len(output_array), 1024 // 2 + 1)

    def test_irfft_float(self):
        input_array = np.random.rand(1024 // 2 + 1) + 1j * np.random.rand(1024 // 2 + 1)
        input_array = input_array.astype(np.complex64)
        output_array = omnidsp_py.irfft_float(input_array)
        self.assertEqual(len(output_array), 1024)

    def test_irfft_double(self):
        input_array = np.random.rand(1024 // 2 + 1) + 1j * np.random.rand(1024 // 2 + 1)
        input_array = input_array.astype(np.complex128)
        output_array = omnidsp_py.irfft_double(input_array)
        self.assertEqual(len(output_array), 1024)


if __name__ == '__main__':
    unittest.main()

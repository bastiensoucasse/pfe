import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath (__file__)))
from Rigid import (
    sitk,
    rigid_registration
)


class TestRigidMethods(unittest.TestCase):
    def test_rigid_1(self):
        thisPath = os.path.dirname(os.path.abspath(__file__))
        fixed_image = os.path.join(thisPath, "test_data", "RegLib_C01_MRMeningioma_1.nrrd")
        moving_image = os.path.join(thisPath, "test_data", "RegLib_C01_MRMeningioma_2.nrrd")

        fixed_image = sitk.ReadImage(fixed_image, sitk.sitkFloat32)
        moving_image = sitk.ReadImage(moving_image, sitk.sitkFloat32)

        parameters = {}
        parameters["metrics"] = "MeanSquares"
        parameters["interpolator"] = "Linear"
        parameters["optimizer"] = "GradientDescent"
        parameters["histogram_bin_count"] = 50
        parameters["sampling_strategy"] = 2
        #seed to avoid randomization
        parameters["sampling_percentage"] = 0.01

        # parameters for gradient optimizer
        parameters["learning_rate"] = 5
        parameters["iterations"] = 100
        parameters["convergence_min_val"] = 1e-6
        parameters["convergence_win_size"] = 10
        # parameters for exhaustive optimizer
        parameters["nb_of_steps"] = None
        parameters["step_length"] = None
        parameters["optimizer_scale"] = None
        # parameters for lbfgs2 optimizer
        parameters["gradient_conv_tol"] = None
        parameters["nb_iter_lbfgs2"] = None
        parameters["max_nb_correction"] = None
        parameters["max_func_eval"] = None

        final_transform = rigid_registration(fixed_image, moving_image, parameters)
        expected_transform = sitk.ReadTransform(os.path.join(thisPath, "test_data", "expected_transform_1.tfm"))
        self.assertIsNotNone(final_transform)
        self.assertEqual(final_transform.GetDimension(), expected_transform.GetDimension())
        self.assertEqual(final_transform.GetNumberOfFixedParameters(), expected_transform.GetNumberOfFixedParameters())
        self.assertEqual(final_transform.GetNumberOfParameters(), expected_transform.GetNumberOfParameters())
        for x, y in zip(final_transform.GetParameters(), expected_transform.GetParameters()):
            self.assertAlmostEqual(x, y, delta=0.01)
        for x, y in zip(final_transform.GetFixedParameters(), expected_transform.GetFixedParameters()):
            self.assertAlmostEqual(x, y, delta=0.01)

    def test_rigid_registration_2(self):
        thisPath = os.path.dirname(os.path.abspath(__file__))
        fixed_image = os.path.join(thisPath, "test_data", "RegLib_C01_MRMeningioma_1.nrrd")
        moving_image = os.path.join(thisPath, "test_data", "RegLib_C01_MRMeningioma_2.nrrd")

        fixed_image = sitk.ReadImage(fixed_image, sitk.sitkFloat32)
        moving_image = sitk.ReadImage(moving_image, sitk.sitkFloat32)

        parameters = {}
        parameters["metrics"] = "MeanSquares"
        parameters["interpolator"] = "Linear"
        parameters["optimizer"] = "Exhaustive"
        parameters["histogram_bin_count"] = 50
        parameters["sampling_strategy"] = 2
        parameters["sampling_percentage"] = 0.01

        # parameters for gradient optimizer
        parameters["learning_rate"] = None
        parameters["iterations"] = None
        parameters["convergence_min_val"] = None
        parameters["convergence_win_size"] = None
        # parameters for exhaustive optimizer
        parameters["nb_of_steps"] = [1, 1, 1, 0, 0, 0]
        parameters["step_length"] = 3.1415
        parameters["optimizer_scale"] = [1, 1, 1, 1, 1, 1]
        # parameters for lbfgs2 optimizer
        parameters["gradient_conv_tol"] = None
        parameters["nb_iter_lbfgs2"] = None
        parameters["max_nb_correction"] = None
        parameters["max_func_eval"] = None

        final_transform = rigid_registration(fixed_image, moving_image, parameters)
        expected_transform = sitk.ReadTransform(os.path.join(thisPath, "test_data", "expected_transform_2.tfm"))
        self.assertIsNotNone(final_transform)
        self.assertEqual(final_transform.GetDimension(), expected_transform.GetDimension())
        self.assertEqual(final_transform.GetNumberOfFixedParameters(), expected_transform.GetNumberOfFixedParameters())
        self.assertEqual(final_transform.GetNumberOfParameters(), expected_transform.GetNumberOfParameters())
        for x, y in zip(final_transform.GetParameters(), expected_transform.GetParameters()):
            self.assertAlmostEqual(x, y, delta=0.01)
        for x, y in zip(final_transform.GetFixedParameters(), expected_transform.GetFixedParameters()):
            self.assertAlmostEqual(x, y, delta=0.01)

    def test_rigid_registration_3(self):
        thisPath = os.path.dirname(os.path.abspath(__file__))
        fixed_image = os.path.join(thisPath, "test_data", "RegLib_C01_MRMeningioma_1.nrrd")
        moving_image = os.path.join(thisPath, "test_data", "RegLib_C01_MRMeningioma_2.nrrd")

        fixed_image = sitk.ReadImage(fixed_image, sitk.sitkFloat32)
        moving_image = sitk.ReadImage(moving_image, sitk.sitkFloat32)

        parameters = {}
        parameters["metrics"] = "MeanSquares"
        parameters["interpolator"] = "Linear"
        parameters["optimizer"] = "LBFGSB"
        parameters["histogram_bin_count"] = 50
        parameters["sampling_strategy"] = 2
        parameters["sampling_percentage"] = 0.01

        # parameters for gradient optimizer
        parameters["learning_rate"] = None
        parameters["iterations"] = None
        parameters["convergence_min_val"] = None
        parameters["convergence_win_size"] = None
        # parameters for exhaustive optimizer
        parameters["nb_of_steps"] = None
        parameters["step_length"] = None
        parameters["optimizer_scale"] = None
        # parameters for lbfgs2 optimizer
        parameters["gradient_conv_tol"] = 1e-5
        parameters["nb_iter_lbfgs2"] = 100
        parameters["max_nb_correction"] = 5
        parameters["max_func_eval"] = 1000

        final_transform = rigid_registration(fixed_image, moving_image, parameters)
        expected_transform = sitk.ReadTransform(os.path.join(thisPath, "test_data", "expected_transform_3.tfm"))
        self.assertIsNotNone(final_transform)
        self.assertEqual(final_transform.GetDimension(), expected_transform.GetDimension())
        self.assertEqual(final_transform.GetNumberOfFixedParameters(), expected_transform.GetNumberOfFixedParameters())
        self.assertEqual(final_transform.GetNumberOfParameters(), expected_transform.GetNumberOfParameters())
        for x, y in zip(final_transform.GetParameters(), expected_transform.GetParameters()):
            self.assertAlmostEqual(x, y, delta=0.01)
        for x, y in zip(final_transform.GetFixedParameters(), expected_transform.GetFixedParameters()):
            self.assertAlmostEqual(x, y, delta=0.01)

if __name__ == '__main__':
    unittest.main()
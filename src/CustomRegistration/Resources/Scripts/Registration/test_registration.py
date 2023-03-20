import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath (__file__)))
from Rigid import (
    sitk,
    rigid_registration
)
from NonRigid import non_rigid_registration
from Utilities import select_metrics, select_interpolator, select_optimizer_and_setup


class TestRigidMethods(unittest.TestCase):
    def setUp(self):
        self.thisPath = os.path.dirname(os.path.abspath(__file__))
        self.fixed_image = os.path.join(self.thisPath, "..", "..","TestData", "RegLib_C01_MRMeningioma_1.nrrd")
        self.moving_image = os.path.join(self.thisPath, "..", "..","TestData", "RegLib_C01_MRMeningioma_2.nrrd")
        self.resampled_image = os.path.join(self.thisPath, "..", "..","TestData", "resampled.nrrd")
        self.affine_image = os.path.join(self.thisPath, "..", "..", "TestData", "affine_reg.nrrd")
        self.fixed_image = sitk.ReadImage(self.fixed_image, sitk.sitkFloat32)
        self.moving_image = sitk.ReadImage(self.moving_image, sitk.sitkFloat32)
        self.resampled_image = sitk.ReadImage(self.resampled_image, sitk.sitkFloat32)
        self.affine_image = sitk.ReadImage(self.affine_image, sitk.sitkFloat32)

    def test_rigid_registration_1(self):
        parameters = {}
        parameters["metrics"] = "MeanSquares"
        parameters["interpolator"] = "Linear"
        parameters["optimizer"] = "Gradient Descent"
        parameters["histogram_bin_count"] = 50
        parameters["sampling_strategy"] = 2
        parameters["sampling_percentage"] = 0.01
        # parameters for gradient optimizer
        parameters["learning_rate"] = 5
        parameters["nb_iteration"] = 100
        parameters["convergence_min_val"] = 1e-6
        parameters["convergence_win_size"] = 10

        final_transform = rigid_registration(self.fixed_image, self.moving_image, parameters)
        expected_transform = sitk.ReadTransform(os.path.join(self.thisPath, "test_data", "expected_transform_1.tfm"))
        self.assertIsNotNone(final_transform)
        self.assertEqual(final_transform.GetDimension(), expected_transform.GetDimension())
        self.assertEqual(final_transform.GetNumberOfFixedParameters(), expected_transform.GetNumberOfFixedParameters())
        self.assertEqual(final_transform.GetNumberOfParameters(), expected_transform.GetNumberOfParameters())
        for x, y in zip(final_transform.GetParameters(), expected_transform.GetParameters()):
            self.assertAlmostEqual(x, y, delta=0.01)
        for x, y in zip(final_transform.GetFixedParameters(), expected_transform.GetFixedParameters()):
            self.assertAlmostEqual(x, y, delta=0.01)

    def test_rigid_registration_2(self):
        parameters = {}
        parameters["metrics"] = "MeanSquares"
        parameters["interpolator"] = "Linear"
        parameters["optimizer"] = "Exhaustive"
        parameters["histogram_bin_count"] = 50
        parameters["sampling_strategy"] = 2
        parameters["sampling_percentage"] = 0.01
        # parameters for exhaustive optimizer
        parameters["nb_of_steps"] = [1, 1, 1, 0, 0, 0]
        parameters["step_length"] = 3.1415
        parameters["optimizer_scale"] = [1, 1, 1, 1, 1, 1]

        final_transform = rigid_registration(self.fixed_image, self.moving_image, parameters)
        expected_transform = sitk.ReadTransform(os.path.join(self.thisPath, "test_data", "expected_transform_2.tfm"))
        self.assertIsNotNone(final_transform)
        self.assertEqual(final_transform.GetDimension(), expected_transform.GetDimension())
        self.assertEqual(final_transform.GetNumberOfFixedParameters(), expected_transform.GetNumberOfFixedParameters())
        self.assertEqual(final_transform.GetNumberOfParameters(), expected_transform.GetNumberOfParameters())
        for x, y in zip(final_transform.GetParameters(), expected_transform.GetParameters()):
            self.assertAlmostEqual(x, y, delta=0.01)
        for x, y in zip(final_transform.GetFixedParameters(), expected_transform.GetFixedParameters()):
            self.assertAlmostEqual(x, y, delta=0.01)

    def test_rigid_registration_3(self):
        parameters = {}
        parameters["metrics"] = "MeanSquares"
        parameters["interpolator"] = "Linear"
        parameters["optimizer"] = "LBFGSB"
        parameters["histogram_bin_count"] = 50
        parameters["sampling_strategy"] = 2
        parameters["sampling_percentage"] = 0.01
        # parameters for lbfgs2 optimizer
        parameters["gradient_conv_tol"] = 1e-5
        parameters["nb_iter_lbfgsb"] = 100
        parameters["max_nb_correction"] = 5
        parameters["max_func_eval"] = 1000

        final_transform = rigid_registration(self.fixed_image, self.moving_image, parameters)
        expected_transform = sitk.ReadTransform(os.path.join(self.thisPath, "test_data", "expected_transform_3.tfm"))
        self.assertIsNotNone(final_transform)
        self.assertEqual(final_transform.GetDimension(), expected_transform.GetDimension())
        self.assertEqual(final_transform.GetNumberOfFixedParameters(), expected_transform.GetNumberOfFixedParameters())
        self.assertEqual(final_transform.GetNumberOfParameters(), expected_transform.GetNumberOfParameters())
        for x, y in zip(final_transform.GetParameters(), expected_transform.GetParameters()):
            self.assertAlmostEqual(x, y, delta=0.01)
        for x, y in zip(final_transform.GetFixedParameters(), expected_transform.GetFixedParameters()):
            self.assertAlmostEqual(x, y, delta=0.01)

    # lbfgsb optimizer not working with scale_factor
    def test_non_rigid_registration_1(self):
        parameters = {}
        parameters["metrics"] = "MeanSquares"
        parameters["interpolator"] = "Linear"
        parameters["algorithm"] = "Non Rigid Bspline"
        parameters["optimizer"] = "LBFGSB"
        parameters["histogram_bin_count"] = 50
        parameters["sampling_strategy"] = 2
        parameters["sampling_percentage"] = 0.01
        # parameters for lbfgs2 optimizer
        parameters["gradient_conv_tol"] = 1e-5
        parameters["nb_iter_lbfgsb"] = 10
        parameters["max_nb_correction"] = 5
        parameters["max_func_eval"] = 1000
        # parameters for bspline
        parameters["transform_domain_mesh_size"] = 2
        parameters["scale_factor"] = None
        parameters["shrink_factor"] = [4, 2, 1]
        parameters["smoothing_sigmas"] =  [2, 1, 0]


        final_transform = non_rigid_registration(self.fixed_image, self.moving_image, parameters)
        expected_transform = sitk.ReadTransform(os.path.join(self.thisPath, "test_data", "expected_transform_4.tfm"))
        self.assertIsNotNone(final_transform)
        self.assertEqual(final_transform.GetDimension(), expected_transform.GetDimension())
        self.assertEqual(final_transform.GetNumberOfFixedParameters(), expected_transform.GetNumberOfFixedParameters())
        self.assertEqual(final_transform.GetNumberOfParameters(), expected_transform.GetNumberOfParameters())
        for x, y in zip(final_transform.GetParameters(), expected_transform.GetParameters()):
            self.assertAlmostEqual(x, y, delta=0.01)
        for x, y in zip(final_transform.GetFixedParameters(), expected_transform.GetFixedParameters()):
            self.assertAlmostEqual(x, y, delta=0.01)

    def test_non_rigid_registration_2(self):
        parameters = {}
        parameters["metrics"] = "MattesMutualInformation"
        parameters["interpolator"] = "Linear"
        parameters["algorithm"] = "Non Rigid Bspline"
        parameters["optimizer"] = "Gradient Descent"
        parameters["histogram_bin_count"] = 50
        parameters["sampling_strategy"] = 2
        parameters["sampling_percentage"] = 0.01
        # parameters for gradient optimizer
        parameters["learning_rate"] = 1
        parameters["nb_iteration"] = 100
        parameters["convergence_min_val"] = 1e-5
        parameters["convergence_win_size"] = 10
        # parameters for bspline
        parameters["transform_domain_mesh_size"] = 1
        parameters["scale_factor"] = [1, 2, 4]
        parameters["shrink_factor"] = [4, 2, 1]
        parameters["smoothing_sigmas"] = [2, 1, 0]

        final_transform = non_rigid_registration(self.fixed_image, self.moving_image, parameters)
        expected_transform = sitk.ReadTransform(os.path.join(self.thisPath, "test_data", "expected_transform_5.tfm"))
        self.assertIsNotNone(final_transform)
        self.assertEqual(final_transform.GetDimension(), expected_transform.GetDimension())
        self.assertEqual(final_transform.GetNumberOfFixedParameters(), expected_transform.GetNumberOfFixedParameters())
        self.assertEqual(final_transform.GetNumberOfParameters(), expected_transform.GetNumberOfParameters())
        
    def test_non_rigid_registration_3(self):
        parameters = {}
        parameters["metrics"] = "MattesMutualInformation"
        parameters["interpolator"] = "Linear"
        parameters["algorithm"] = "Non Rigid Bspline"
        parameters["optimizer"] = "LBFGSB"
        parameters["histogram_bin_count"] = 50
        parameters["sampling_strategy"] = 2
        parameters["sampling_percentage"] = 0.01

        # parameters for gradient optimizer
        parameters["learning_rate"] = None
        parameters["nb_iteration"] = None
        parameters["convergence_min_val"] = None
        parameters["convergence_win_size"] = None
        # parameters for exhaustive optimizer
        parameters["nb_of_steps"] = None
        parameters["step_length"] = None
        parameters["optimizer_scale"] = None
        # parameters for lbfgs2 optimizer
        parameters["gradient_conv_tol"] = 1e-5
        parameters["nb_iter_lbfgsb"] = 100
        parameters["max_nb_correction"] = 5
        parameters["max_func_eval"] = 1000
        # parameters for bspline
        parameters["transform_domain_mesh_size"] = 1
        parameters["scale_factor"] = [1, 2, 4]
        parameters["shrink_factor"] = [4, 2, 1]
        parameters["smoothing_sigmas"] = [4, 2, 1]
        # parameters for demons
        parameters["demons_nb_iter"] = None
        parameters["demons_std_dev"] = None
        self.assertRaises(RuntimeError, non_rigid_registration, self.fixed_image, self.moving_image, parameters)

    def test_demons_registration_1(self):
        parameters = {}
        parameters["metrics"] = "MattesMutualInformation"
        parameters["interpolator"] = "Linear"
        parameters["algorithm"] = "Demons"
        parameters["optimizer"] = "Gradient Descent"
        parameters["histogram_bin_count"] = 50
        parameters["sampling_strategy"] = 2
        parameters["sampling_percentage"] = 0.01
        # parameters for bspline
        parameters["transform_domain_mesh_size"] = 1
        parameters["scale_factor"] = [1, 2, 4]
        parameters["shrink_factor"] = [4, 2, 1]
        parameters["smoothing_sigmas"] = [4, 2, 1]
        # parameters for demons
        parameters["demons_nb_iter"] = 50
        parameters["demons_std_dev"] = 1

        final_transform = non_rigid_registration(self.fixed_image, self.moving_image, parameters)
        #because images are not resampled, or registrated.
        self.assertIsNone(final_transform)

    def test_demons_registration_2(self):
        parameters = {}
        parameters["interpolator"] = "Linear"
        parameters["algorithm"] = "Demons"
        # parameters for demons
        parameters["demons_nb_iter"] = 50
        parameters["demons_std_dev"] = 1

        final_transform = non_rigid_registration(self.fixed_image, self.resampled_image, parameters)
        self.assertIsNotNone(final_transform)

        demons = sitk.DemonsRegistrationFilter()
        demons.SetNumberOfIterations(50)
        demons.SetStandardDeviations(1.0)
        displacementField = demons.Execute(self.fixed_image, self.resampled_image)
        expected_transform = sitk.DisplacementFieldTransform(displacementField)

        self.assertEqual(final_transform.GetDimension(), expected_transform.GetDimension())
        self.assertEqual(final_transform.GetNumberOfFixedParameters(), expected_transform.GetNumberOfFixedParameters())
        self.assertEqual(final_transform.GetNumberOfParameters(), expected_transform.GetNumberOfParameters())
        for x, y in zip(final_transform.GetFixedParameters(), expected_transform.GetFixedParameters()):
            self.assertAlmostEqual(x, y, delta=0.01)

    def test_demons_registration_3(self):
        parameters = {}
        parameters["algorithm"] = "DiffeomorphicDemons"
        parameters["demons_nb_iter"] = 50
        parameters["demons_std_dev"] = 1.0

        final_transform = non_rigid_registration(self.fixed_image, self.resampled_image, parameters)
        #diffeomorphic demons needs same spacing for both images, hence a rigid or affige reg is required before use.
        self.assertIsNone(final_transform)

        final_transform = non_rigid_registration(self.fixed_image, self.affine_image, parameters)
        demons = sitk.DiffeomorphicDemonsRegistrationFilter()
        demons.SetNumberOfIterations(50)
        demons.SetStandardDeviations(1.0)
        displacementField = demons.Execute(self.fixed_image, self.affine_image)
        expected_transform = sitk.DisplacementFieldTransform(displacementField)

        self.assertEqual(final_transform.GetDimension(), expected_transform.GetDimension())
        self.assertEqual(final_transform.GetNumberOfFixedParameters(), expected_transform.GetNumberOfFixedParameters())
        self.assertEqual(final_transform.GetNumberOfParameters(), expected_transform.GetNumberOfParameters())
        for x, y in zip(final_transform.GetParameters(), expected_transform.GetParameters()):
            self.assertAlmostEqual(x, y, delta=1)
        for x, y in zip(final_transform.GetFixedParameters(), expected_transform.GetFixedParameters()):
            self.assertAlmostEqual(x, y, delta=0.01)

    def test_demons_registration_4(self):
        parameters = {}
        parameters["algorithm"] = "FastSymmetricForcesDemons"
        parameters["demons_nb_iter"] = 25
        parameters["demons_std_dev"] = 1.0

        final_transform = non_rigid_registration(self.fixed_image, self.affine_image, parameters)
        #diffeomorphic demons needs same spacing for both images, hence a rigid or affige reg is required before use.
        self.assertIsNotNone(final_transform)

        demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
        demons.SetNumberOfIterations(25)
        demons.SetStandardDeviations(1.0)
        displacementField = demons.Execute(self.fixed_image, self.affine_image)
        expected_transform = sitk.DisplacementFieldTransform(displacementField)

        self.assertEqual(final_transform.GetDimension(), expected_transform.GetDimension())
        self.assertEqual(final_transform.GetNumberOfFixedParameters(), expected_transform.GetNumberOfFixedParameters())
        self.assertEqual(final_transform.GetNumberOfParameters(), expected_transform.GetNumberOfParameters())
        for x, y in zip(final_transform.GetParameters(), expected_transform.GetParameters()):
            self.assertAlmostEqual(x, y, delta=1)
        for x, y in zip(final_transform.GetFixedParameters(), expected_transform.GetFixedParameters()):
            self.assertAlmostEqual(x, y, delta=0.01)

    def test_demons_registration_5(self):
        parameters = {}
        parameters["algorithm"] = "SymmetricForcesDemons"
        parameters["demons_nb_iter"] = 25
        parameters["demons_std_dev"] = 1.0

        final_transform = non_rigid_registration(self.fixed_image, self.affine_image, parameters)
        #diffeomorphic demons needs same spacing for both images, hence a rigid or affige reg is required before use.
        self.assertIsNotNone(final_transform)

        demons = sitk.SymmetricForcesDemonsRegistrationFilter()
        demons.SetNumberOfIterations(25)
        demons.SetStandardDeviations(1.0)
        displacementField = demons.Execute(self.fixed_image, self.affine_image)
        expected_transform = sitk.DisplacementFieldTransform(displacementField)

        self.assertEqual(final_transform.GetDimension(), expected_transform.GetDimension())
        self.assertEqual(final_transform.GetNumberOfFixedParameters(), expected_transform.GetNumberOfFixedParameters())
        self.assertEqual(final_transform.GetNumberOfParameters(), expected_transform.GetNumberOfParameters())
        for x, y in zip(final_transform.GetParameters(), expected_transform.GetParameters()):
            self.assertAlmostEqual(x, y, delta=0.01)
        for x, y in zip(final_transform.GetFixedParameters(), expected_transform.GetFixedParameters()):
            self.assertAlmostEqual(x, y, delta=0.01)

    def test_select_metrics(self):

        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsJointHistogramMutualInformation()
        select_metrics(R, 50, "MattesMutualInformation")
        R.SetMetricSamplingPercentage(0.001, seed=10)
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetOptimizerAsGradientDescent(learningRate=5,
            numberOfIterations=100,
            convergenceMinimumValue=1e-5,
            convergenceWindowSize=10)
        initial_transform = sitk.CenteredTransformInitializer(
            self.fixed_image,
            self.moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        R.SetInitialTransform(initial_transform)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerScalesFromIndexShift()

        outTx = R.Execute(self.fixed_image, self.moving_image)

        R2 = sitk.ImageRegistrationMethod()
        R2.SetMetricAsMattesMutualInformation(50)
        R2.SetMetricSamplingPercentage(0.001, seed=10)
        R2.SetMetricSamplingStrategy(R2.RANDOM)
        R2.SetOptimizerAsGradientDescent(learningRate=5,
            numberOfIterations=100,
            convergenceMinimumValue=1e-5,
            convergenceWindowSize=10)
        initial_transform = sitk.CenteredTransformInitializer(
            self.fixed_image,
            self.moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        R2.SetInitialTransform(initial_transform)
        R2.SetInterpolator(sitk.sitkLinear)
        R2.SetOptimizerScalesFromIndexShift()
        outTx2 = R2.Execute(self.fixed_image, self.moving_image)

        for x, y in zip(outTx.GetParameters(), outTx2.GetParameters()):
            self.assertEqual(x, y, delta=0.0001)
        for x, y in zip(outTx.GetFixedParameters(), outTx2.GetFixedParameters()):
            self.assertEqual(x, y, delta=0.0001)

    def test_select_interpolator(self):
        R = sitk.ImageRegistrationMethod()
        select_interpolator(R, "BSpline1")
        self.assertEqual(R.GetInterpolator(), sitk.sitkBSplineResamplerOrder1)
        select_interpolator(R, "BSpline2")
        self.assertEqual(R.GetInterpolator(), sitk.sitkBSplineResamplerOrder2)
        select_interpolator(R, "Gaussian")
        self.assertEqual(R.GetInterpolator(), sitk.sitkGaussian)
        select_interpolator(R, "Linear")
        self.assertEqual(R.GetInterpolator(), sitk.sitkLinear)
        select_interpolator(R, "Nearest Neighbor")
        self.assertEqual(R.GetInterpolator(), sitk.sitkNearestNeighbor)

    def test_select_optimizer_1(self):
        R = sitk.ImageRegistrationMethod()
        input = {}
        input["optimizer"] = "Gradient Descent"
        input["learning_rate"] = 5
        input["nb_iteration"] = 100
        input["convergence_min_val"] = 1e-5
        input["convergence_win_size"] = 10
        select_optimizer_and_setup(R, input)

        R.SetMetricAsMeanSquares()
        R.SetMetricSamplingPercentage(0.001, seed=15)
        R.SetMetricSamplingStrategy(R.RANDOM)
        initial_transform = sitk.CenteredTransformInitializer(
            self.fixed_image,
            self.moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        R.SetInitialTransform(initial_transform)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerScalesFromIndexShift()

        output_transform = R.Execute(self.fixed_image, self.moving_image)


        R2 = sitk.ImageRegistrationMethod()
        R2.SetMetricAsMeanSquares()
        R2.SetMetricSamplingPercentage(0.001, seed=15)
        R2.SetMetricSamplingStrategy(R2.RANDOM)
        R2.SetOptimizerAsGradientDescent(learningRate=5,
            numberOfIterations=100,
            convergenceMinimumValue=1e-5,
            convergenceWindowSize=10)
        initial_transform = sitk.CenteredTransformInitializer(
            self.fixed_image,
            self.moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        R2.SetInitialTransform(initial_transform)
        R2.SetInterpolator(sitk.sitkLinear)
        R2.SetOptimizerScalesFromIndexShift()
        expected_transform = R2.Execute(self.fixed_image, self.moving_image)

        for x, y in zip(output_transform.GetParameters(), expected_transform.GetParameters()):
            self.assertAlmostEqual(x, y, delta=0.01)
        for x, y in zip(output_transform.GetFixedParameters(), expected_transform.GetFixedParameters()):
            self.assertAlmostEqual(x, y, delta=0.01)

    def test_select_optimizer_2(self):
        R = sitk.ImageRegistrationMethod()
        input = {}
        input["optimizer"] = "Exhaustive"
        input["nb_of_steps"] = [1, 1, 1, 0, 0, 0]
        input["step_length"] = 1
        input["optimizer_scale"] = [1,1,1,1,1,1]
        select_optimizer_and_setup(R, input)

        R.SetMetricAsMeanSquares()
        R.SetMetricSamplingPercentage(0.001, seed=15)
        R.SetMetricSamplingStrategy(R.RANDOM)
        initial_transform = sitk.CenteredTransformInitializer(
            self.fixed_image,
            self.moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        R.SetInitialTransform(initial_transform)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerScalesFromIndexShift()

        output_transform = R.Execute(self.fixed_image, self.moving_image)


        R2 = sitk.ImageRegistrationMethod()
        R2.SetMetricAsMeanSquares()
        R2.SetMetricSamplingPercentage(0.001, seed=15)
        R2.SetMetricSamplingStrategy(R2.RANDOM)
        R2.SetOptimizerAsExhaustive(numberOfSteps=[1, 1, 1, 0, 0, 0], stepLength=1)
        R2.SetOptimizerScales([1,1,1,1,1,1])
        initial_transform = sitk.CenteredTransformInitializer(
            self.fixed_image,
            self.moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        R2.SetInitialTransform(initial_transform)
        R2.SetInterpolator(sitk.sitkLinear)
        R2.SetOptimizerScalesFromIndexShift()
        expected_transform = R2.Execute(self.fixed_image, self.moving_image)

        for x, y in zip(output_transform.GetParameters(), expected_transform.GetParameters()):
            self.assertAlmostEqual(x, y, delta=0.01)
        for x, y in zip(output_transform.GetFixedParameters(), expected_transform.GetFixedParameters()):
            self.assertAlmostEqual(x, y, delta=0.01)

    def test_select_optimizer_3(self):
        R = sitk.ImageRegistrationMethod()
        input = {}
        input["optimizer"] = "LBFGSB"
        input["gradient_conv_tol"] = 1e-2
        input["nb_iter_lbfgsb"] = 100
        input["max_nb_correction"] = 5
        input["max_func_eval"] = 1000
        select_optimizer_and_setup(R, input)

        R.SetMetricAsMeanSquares()
        R.SetMetricSamplingPercentage(0.001, seed=15)
        R.SetMetricSamplingStrategy(R.RANDOM)
        initial_transform = sitk.CenteredTransformInitializer(
            self.fixed_image,
            self.moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        R.SetInitialTransform(initial_transform)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerScalesFromIndexShift()

        output_transform = R.Execute(self.fixed_image, self.moving_image)


        R2 = sitk.ImageRegistrationMethod()
        R2.SetMetricAsMeanSquares()
        R2.SetMetricSamplingPercentage(0.001, seed=15)
        R2.SetMetricSamplingStrategy(R2.RANDOM)
        R2.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-2,
        numberOfIterations=100, 
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000)
        initial_transform = sitk.CenteredTransformInitializer(
            self.fixed_image,
            self.moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        R2.SetInitialTransform(initial_transform)
        R2.SetInterpolator(sitk.sitkLinear)
        R2.SetOptimizerScalesFromIndexShift()
        expected_transform = R2.Execute(self.fixed_image, self.moving_image)

        for x, y in zip(output_transform.GetParameters(), expected_transform.GetParameters()):
            self.assertAlmostEqual(x, y, delta=0.0001)
        for x, y in zip(output_transform.GetFixedParameters(), expected_transform.GetFixedParameters()):
            self.assertAlmostEqual(x, y, delta=0.0001)

    def test_select_optimizer_4(self):
        R = sitk.ImageRegistrationMethod()
        input = {}
        input["optimizer"] = "LBFGSB"
        input["gradient_conv_tol"] = None
        input["nb_iter_lbfgsb"] = 100
        input["max_nb_correction"] = 5
        input["max_func_eval"] = 1000
        self.assertRaises(TypeError, select_optimizer_and_setup, R, input)
        input["gradient_conv_tol"] = 1e-2
        input["nb_iter_lbfgsb"] = -5
        self.assertRaises(OverflowError, select_optimizer_and_setup, R, input)
        input["nb_iter_lbfgsb"] = 1
        input["max_nb_correction"] = -2
        self.assertRaises(OverflowError, select_optimizer_and_setup, R, input)
        input["max_nb_correction"] = 0
        input["max_func_eval"] = 1e-5
        self.assertRaises(TypeError, select_optimizer_and_setup, R, input)
        input["optimizer"]="mqfldk"
        self.assertRaises(ValueError, select_optimizer_and_setup, R, input)

if __name__ == '__main__':
    testClass = TestRigidMethods()
    testClass.setUp()
    testClass.test_demons_registration_5()
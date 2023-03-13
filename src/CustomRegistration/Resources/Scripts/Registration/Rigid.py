"""
Rigid registration algorithms and testing scripts.
"""

import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath (__file__)))
import SimpleITK as sitk
import Utilities as util

def rigid_registration(fixed_image, moving_image, parameters):
    metrics_name = parameters["metrics"]
    interpolator_name = parameters["interpolator"]
    optimizer_name = parameters["optimizer"]
    bin_count = parameters["histogram_bin_count"]
    sampling_strat = parameters["sampling_strategy"]
    sampling_perc = parameters["sampling_percentage"]

    # parameters for gradient optimizer
    learning_rate = parameters["learning_rate"]
    nb_iteration = parameters["iterations"]
    convergence_min_val = parameters["convergence_min_val"]
    convergence_win_size = parameters["convergence_win_size"]
    # parameters for exhaustive optimizer
    nb_of_steps = parameters["nb_of_steps"]
    step_length = parameters["step_length"]
    optimizer_scale = parameters["optimizer_scale"]
    # parameters for lbfgs2 optimizer
    solution_acc = parameters["solution_accuracy"]
    nb_iter_lbfgs2 = parameters["nb_iter_lbfgs2"]
    delta_conv_tol = parameters["delta_convergence_tolerance"]

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    R = sitk.ImageRegistrationMethod()
    R.SetMetricSamplingStrategy(sampling_strat)
    R.SetMetricSamplingPercentage(sampling_perc)
    util.select_metrics(R, bin_count, metrics_name)
    util.select_interpolator(R, interpolator_name)
    util.select_optimizer_and_setup(
        R,
        optimizer_name,
        learning_rate,
        nb_iteration,
        convergence_min_val,
        convergence_win_size,
        nb_of_steps,
        step_length,
        optimizer_scale,
        solution_acc,
        nb_iter_lbfgs2,
        delta_conv_tol,
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = R.Execute(fixed_image, moving_image)
    print("-------")
    print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    print(f" Iteration: {R.GetOptimizerIteration()}")
    print(f" Metric value: {R.GetMetricValue()}")

    return final_transform

def test_rigid_registration():
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
    parameters["sampling_percentage"] = 0.001

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
    parameters["solution_accuracy"] = None
    parameters["nb_iter_lbfgs2"] = None
    parameters["delta_convergence_tolerance"] = None

    final_transform = rigid_registration(fixed_image, moving_image, parameters)
    expected_transform = sitk.ReadTransform(os.path.join(thisPath, "test_data", "expected_transform_1.tfm"))
    assert final_transform
    assert final_transform.GetDimension() == expected_transform.GetDimension()
    assert final_transform.GetNumberOfFixedParameters() == expected_transform.GetNumberOfFixedParameters()
    assert final_transform.GetNumberOfParameters() == expected_transform.GetNumberOfParameters()

    print("rigid test 1 passed !")

if __name__ == "__main__":
    if "--test" in sys.argv:
        test_rigid_registration()
        sys.exit(0)
    pickledInput = sys.stdin.buffer.read()
    input = pickle.loads(pickledInput)
    fixed_image = input["fixed_image"]
    moving_image = input["moving_image"]
    # user inputs
    parameters = input["parameters"]
    final_transform = rigid_registration(fixed_image, moving_image, parameters)

    resampled = sitk.Resample(
        moving_image,
        fixed_image,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID(),
    )

    output = {}
    output["image_resampled"] = resampled
    output["volume_name"] = parameters["volume_name"]
    sys.stdout.buffer.write(pickle.dumps(output))
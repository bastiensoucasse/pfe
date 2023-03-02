"""
Rigid registration algorithms and testing scripts.
"""

import pickle
import sys

import SimpleITK as sitk
import Utilities as util

pickledInput = sys.stdin.buffer.read()
input = pickle.loads(pickledInput)
fixed_image = input["fixed_image"]
moving_image = input["moving_image"]
# user inputs
parameters = input["parameters"]
volume_name = parameters["volume_name"]
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


def main():
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bin_count)
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
    output["pixelID"] = fixed_image.GetPixelID()

    sys.stdout.buffer.write(pickle.dumps(output))


main()

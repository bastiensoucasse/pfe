"""
Rigid registration algorithms and testing scripts.
"""

import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath (__file__)))
import SimpleITK as sitk
import Utilities as util

error = []
optimizer_end_results = []

def rigid_registration(fixed_image, moving_image, parameters) -> sitk.Transform:
    """
    Perfoms a rigid or affine registration.

    Parameters:
        fixed_image: the reference image.
        moving_image: the image to registrate.
        parameters: a dictionary taht contains all sorts of user parameters (metrics chosed, registration algorithm...)

    Return : the result of the registration, a transform
    """
    algorithm = parameters["algorithm"]

    if algorithm == "Affine":
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, 
            moving_image, 
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY)
    else:
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY)

    metrics_name = parameters["metrics"]
    interpolator_name = parameters["interpolator"]
    bin_count = parameters["histogram_bin_count"]
    sampling_strat = parameters["sampling_strategy"]
    sampling_perc = parameters["sampling_percentage"]

    R = sitk.ImageRegistrationMethod()
    R.SetMetricSamplingStrategy(sampling_strat)
    R.SetMetricSamplingPercentage(sampling_perc, seed=10)
    util.select_metrics(R, metrics_name, bin_count)
    util.select_interpolator(R, interpolator_name)
    util.select_optimizer_and_setup(R, parameters)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = R.Execute(fixed_image, moving_image)
    optimizer_end_results.append(R.GetOptimizerStopConditionDescription())
    optimizer_end_results.append(R.GetOptimizerIteration())
    optimizer_end_results.append(R.GetMetricValue())

    return final_transform

if __name__ == "__main__":
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
    output["error"] = "\n".join(error)
    output["stop_condition"] = optimizer_end_results[0]
    output["nb_iteration"] = optimizer_end_results[1]
    output["metric_value"] = optimizer_end_results[2]
    sys.stdout.buffer.write(pickle.dumps(output))
"""
Non-rigid registration algorithms and testing scripts.
"""

import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath (__file__)))
import SimpleITK as sitk
import Utilities as util

error = []
optimizer_end_results = []

def non_rigid_registration(fixed_image, moving_image, parameters) -> sitk.Transform:
    """
    Performs a non rigid registration : BSpline or Demons

    Parameters:
        fixed_image: the reference image.
        moving_image: the image to registrate.
        paramters: a dictionary taht contains all sorts of user parameters (metrics chosed, registration algorithm...)

    Return : the result of the registration, a transform
    """
    algorithm = parameters["algorithm"]

    if "Demons" in algorithm:
        demons = util.get_demons_algorithm(algorithm)
        demons.SetNumberOfIterations(NumberOfIterations=parameters["demons_nb_iter"])
        demons.SetStandardDeviations(parameters["demons_std_dev"])
        toDisplacementFilter = sitk.TransformToDisplacementFieldFilter()
        toDisplacementFilter.SetReferenceImage(fixed_image)
        initialTransform = sitk.CenteredTransformInitializer(fixed_image, 
            moving_image, 
            sitk.AffineTransform(3), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY)
        displacementField = toDisplacementFilter.Execute(initialTransform)
        try:
            displacementField = demons.Execute(fixed_image, moving_image, displacementField)
            outTx = sitk.DisplacementFieldTransform(displacementField)
        except RuntimeError as rt:
            error.append(str(rt))
            error.append("Try resampling or registration using affine before using any demons algorithm")
            outTx = None
        finally:
            return outTx

    else:
        metrics_name = parameters["metrics"]
        interpolator_name = parameters["interpolator"]
        bin_count = parameters["histogram_bin_count"]
        sampling_strat = parameters["sampling_strategy"]
        sampling_perc = parameters["sampling_percentage"]
        # parameters for bspline
        transform_domain_mesh_size = parameters["transform_domain_mesh_size"]
        scale_factor = parameters["scale_factor"]
        shrink_factor = parameters["shrink_factor"]
        smoothing_sigmas = parameters["smoothing_sigmas"]
        transformDomainMeshSize = [transform_domain_mesh_size] * fixed_image.GetDimension()
        tx = sitk.BSplineTransformInitializer(fixed_image, transformDomainMeshSize)

        R = sitk.ImageRegistrationMethod()
        R.SetMetricSamplingStrategy(sampling_strat)
        R.SetMetricSamplingPercentage(sampling_perc, seed=10)
        util.select_metrics(R, bin_count, metrics_name)
        util.select_interpolator(R, interpolator_name)
        util.select_optimizer_and_setup(R, parameters)
        if scale_factor != None:
            R.SetInitialTransformAsBSpline(tx, inPlace=True, scaleFactors=scale_factor)
        else:
            R.SetInitialTransformAsBSpline(tx, inPlace=True)
            
        R.SetShrinkFactorsPerLevel(shrink_factor)
        R.SetSmoothingSigmasPerLevel(smoothing_sigmas)
        R.SetOptimizerScalesFromPhysicalShift()
        outTx = R.Execute(fixed_image, moving_image)
        optimizer_end_results.append(R.GetOptimizerStopConditionDescription())
        optimizer_end_results.append(R.GetOptimizerIteration())
        optimizer_end_results.append(R.GetMetricValue())
        return outTx

if __name__ == "__main__":
    pickledInput = sys.stdin.buffer.read()
    input = pickle.loads(pickledInput)
    fixed_image = input["fixed_image"]
    moving_image = input["moving_image"]
    # user inputs
    parameters = input["parameters"]
    final_transform = non_rigid_registration(fixed_image, moving_image, parameters)
    resampled = None
    if final_transform != None:
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
    if "Demons" in parameters["algorithm"]:
        output["stop_condition"] = ""
        output["nb_iteration"] = ""
        output["metric_value"] = ""
    else:
        output["stop_condition"] = optimizer_end_results[0]
        output["nb_iteration"] = optimizer_end_results[1]
        output["metric_value"] = optimizer_end_results[2]
    sys.stdout.buffer.write(pickle.dumps(output))
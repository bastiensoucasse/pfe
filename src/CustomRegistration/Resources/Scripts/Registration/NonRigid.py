"""
Non-rigid registration algorithms and testing scripts.
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
algorithm = parameters["algorithm"]
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
# parameters for bspline
transform_domain_mesh_size = parameters["transform_domain_mesh_size"]
scale_factor = parameters["scale_factor"]
shrink_factor = parameters["shrink_factor"]
smoothing_sigmas = parameters["smoothing_sigmas"]
# parameters for demons
demons_nb_iter = parameters["demons_nb_iter"]
demons_std_dev = parameters["demons_std_dev"]


def main():
    if algorithm == "Non Rigid Demons":
        demons = sitk.DemonsRegistrationFilter()
        demons.SetNumberOfIterations(demons_nb_iter)
        demons.SetStandardDeviations(demons_std_dev)
        displacementField = demons.Execute(fixed_image, moving_image)
        outTx = sitk.DisplacementFieldTransform(displacementField)
    else:
        transformDomainMeshSize = [transform_domain_mesh_size] * fixed_image.GetDimension()
        tx = sitk.BSplineTransformInitializer(fixed_image, transformDomainMeshSize)

        R = sitk.ImageRegistrationMethod()
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

        R.SetInitialTransformAsBSpline(tx, inPlace=True, scaleFactors=scale_factor)
        R.SetShrinkFactorsPerLevel(shrink_factor)
        R.SetSmoothingSigmasPerLevel(smoothing_sigmas)
        outTx = R.Execute(fixed_image, moving_image)


    resampled = sitk.Resample(
        moving_image,
        fixed_image,
        outTx,
        sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID(),
    )

    output = {}
    output["image_resampled"] = resampled
    output["volume_name"] = volume_name

    sys.stdout.buffer.write(pickle.dumps(output))


main()

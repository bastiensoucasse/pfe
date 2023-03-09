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
    transformDomainMeshSize = [2] * fixed_image.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed_image, transformDomainMeshSize)

    print(f"Initial Number of Parameters: {tx.GetNumberOfParameters()}")

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

    R.SetInitialTransformAsBSpline(tx, inPlace=True, scaleFactors=[1, 2, 5])
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([4, 2, 1])

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
